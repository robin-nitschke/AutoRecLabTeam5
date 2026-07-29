"""Re-execute the evaluated Run-5 node and cross-check its numbers.

This script does three things the original study did not do:

  (1) Level 2 -- it re-runs the *exact* pipeline of the evaluated Run-5 node
      (workspace/run5/runfile_iter9_bestnode.py, tree-search iteration 9,
      reviewer score 72/100) in a separate process and compares the new
      metrics with those reported in
      workspace/run5/working/seed_sensitivity_results.csv.

  (2) Level 3 -- it recomputes Precision@k and nDCG@k with an independent
      implementation (a plain per-user loop, not the node's vectorised code)
      and compares that against both the re-run and the reported values.

  (3) Counterfactual -- it repeats every ItemKNN cell with the configuration
      LensKit 0.14.4 documents for implicit feedback (feedback='implicit',
      i.e. aggregate='sum', use_ratings=False) to quantify the effect of the
      misconfiguration found in the evaluated node.

It additionally checks the split for leakage (train/test disjoint per user)
and that the candidate sets exclude items seen in training.

Run from the repository root:

    .venv/Scripts/python.exe scripts/rerun_selected_node.py --datasets ml100k lastfm
    .venv/Scripts/python.exe scripts/rerun_selected_node.py --datasets amazon_videogames --seeds 1

Results are appended to scripts/rerun_results.csv after every cell, so a
partial run still yields usable output.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from lenskit import batch
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'workspace')
OUT_CSV = os.path.join(REPO, 'scripts', 'rerun_results.csv')

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]
TOPN = max(KS)


# ---------------------------------------------------------------------------
# Verbatim from workspace/run5/runfile_iter9_bestnode.py (the evaluated node).
# Do not "improve" these -- the point is to re-execute the node's own logic.
# ---------------------------------------------------------------------------
def k_core_filter(df, user_col='user', item_col='item', min_uc=5, min_ic=5):
    df = df[[user_col, item_col]].drop_duplicates().copy()
    changed = True
    while changed and len(df):
        n0 = len(df)
        uc = df[user_col].value_counts()
        ic = df[item_col].value_counts()
        df = df[df[user_col].isin(uc[uc >= min_uc].index)]
        df = df[df[item_col].isin(ic[ic >= min_ic].index)]
        changed = len(df) != n0
    df = df.reset_index(drop=True)
    df.columns = ['user', 'item']
    return df


def load_ml100k(path='u.data'):
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] > 3][['user', 'item']].drop_duplicates()
    return k_core_filter(df)


def load_amazon(path='VideoGames.csv'):
    try:
        df = pd.read_csv(path)
        cols = list(df.columns)
        low = {c.lower(): c for c in cols}
        if {'userid', 'productid', 'score'}.issubset(low):
            df = df[[low['userid'], low['productid'], low['score']]].copy()
            df.columns = ['user', 'item', 'rating']
        elif {'user', 'item', 'rating'}.issubset(low):
            df = df[[low['user'], low['item'], low['rating']]].copy()
            df.columns = ['user', 'item', 'rating']
        elif len(cols) >= 3:
            df = df.iloc[:, :3].copy()
            df.columns = ['user', 'item', 'rating']
        else:
            raise ValueError('Unexpected Amazon file format')
    except Exception:
        df = pd.read_csv(path, header=None,
                         names=['user', 'item', 'rating', 'timestamp'],
                         usecols=[0, 1, 2])
    df = df[df['rating'] > 3][['user', 'item']].drop_duplicates()
    return k_core_filter(df)


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    df = pd.read_csv(path, sep='\t')
    cols = {c.lower(): c for c in df.columns}
    user_col = cols.get('userid', df.columns[0])
    item_col = cols.get('artistid', df.columns[1])
    df = df[[user_col, item_col]].copy()
    df.columns = ['user', 'item']
    return k_core_filter(df)


def user_holdout_split(df, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, g in df.groupby('user', sort=False):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        n_test = max(1, int(np.ceil(n * test_frac)))
        idx = np.arange(n)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True
        train_parts.append(g.iloc[~mask])
        test_parts.append(g.iloc[mask])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    train_users = set(train['user'].unique())
    train_items = set(train['item'].unique())
    test = test[test['user'].isin(train_users)
                & test['item'].isin(train_items)].reset_index(drop=True)
    return train.reset_index(drop=True), test


def build_candidate_map(train, users, items):
    all_items = set(items)
    seen = train.groupby('user')['item'].agg(set).to_dict()
    return {u: list(all_items - seen.get(u, set())) for u in users}


def ensure_rank(recs):
    recs = recs.copy()
    if 'score' in recs.columns:
        recs = recs.sort_values(['user', 'score'], ascending=[True, False])
    elif 'rank' in recs.columns:
        recs = recs.sort_values(['user', 'rank'])
    else:
        recs = recs.sort_values(['user', 'item'])
    if 'rank' not in recs.columns:
        recs['rank'] = recs.groupby('user').cumcount() + 1
    return recs.reset_index(drop=True)


def node_metrics(recs, test, ks=(1, 5, 10)):
    """The node's own vectorised metric code."""
    truth = test.groupby('user')['item'].agg(set).to_dict()
    users = sorted(truth.keys(), key=lambda x: str(x))
    recs = ensure_rank(recs[recs['user'].isin(users)][['user', 'item', 'rank']].copy())
    gt = test[['user', 'item']].drop_duplicates().copy()
    gt['rel'] = 1
    merged = recs.merge(gt, on=['user', 'item'], how='left')
    merged['rel'] = merged['rel'].fillna(0).astype(int)
    rows = []
    for k in ks:
        mk = merged[merged['rank'] <= k].copy()
        prec_u = mk.groupby('user')['rel'].sum().reindex(users, fill_value=0) / k
        mk['gain'] = mk['rel'] / np.log2(mk['rank'] + 1)
        dcg_u = mk.groupby('user')['gain'].sum().reindex(users, fill_value=0)
        idcg = pd.Series({u: np.sum(1 / np.log2(np.arange(2, min(len(truth[u]), k) + 2)))
                          for u in users})
        ndcg_u = (dcg_u / idcg.replace(0, np.nan)).fillna(0)
        rows.append({'k': k, 'Precision': float(prec_u.mean()), 'nDCG': float(ndcg_u.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Independent metric implementation (level 3): plain per-user loop.
# Deliberately written differently from the node's vectorised version.
# ---------------------------------------------------------------------------
def independent_metrics(recs, test, ks=(1, 5, 10)):
    truth = test.groupby('user')['item'].agg(set).to_dict()
    ranked = ensure_rank(recs)
    by_user = {u: g.sort_values('rank')['item'].tolist()
               for u, g in ranked.groupby('user')}
    out = {}
    for k in ks:
        precs, ndcgs = [], []
        for u in sorted(truth.keys(), key=lambda x: str(x)):
            rel = truth[u]
            topk = by_user.get(u, [])[:k]
            precs.append(sum(1 for it in topk if it in rel) / k)
            dcg = sum(1.0 / np.log2(r + 2) for r, it in enumerate(topk) if it in rel)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        out[f'Precision@{k}'] = float(np.mean(precs))
        out[f'nDCG@{k}'] = float(np.mean(ndcgs))
    return out


LOADERS = {
    'ml100k': (load_ml100k, 'u.data'),
    'amazon_videogames': (load_amazon, 'VideoGames.csv'),
    'lastfm': (load_lastfm, 'UserTaggedArtists-timestamps.dat'),
}

# (label, factory) -- 'node' reproduces the evaluated node, 'implicit' is the
# counterfactual using the configuration LensKit documents for implicit data.
ALGOS = {
    'ALS': ('node', lambda: ImplicitMF(features=50, iterations=15, weight=40,
                                       method='cg', use_ratings=False)),
    'ItemKNN': ('node', lambda: ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6,
                                         center=False)),
    'Pop': ('node', lambda: Popular()),
    'ItemKNN_implicit': ('counterfactual',
                         lambda: ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6,
                                          feedback='implicit')),
}


def append_row(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(OUT_CSV)
    df.to_csv(OUT_CSV, mode='a', header=header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['ml100k', 'lastfm'],
                    choices=list(LOADERS))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--algos', nargs='+', default=list(ALGOS), choices=list(ALGOS))
    args = ap.parse_args()

    os.chdir(DATA)  # the node's loaders use bare filenames
    reported = pd.read_csv(os.path.join(REPO, 'workspace', 'run5', 'working',
                                        'seed_sensitivity_results.csv'))

    for ds_name in args.datasets:
        loader, path = LOADERS[ds_name]
        t0 = time.time()
        data = loader(path)
        print(f'[{ds_name}] 5-core: {len(data):,} interactions, '
              f'{data.user.nunique():,} users, {data.item.nunique():,} items '
              f'({time.time() - t0:.0f}s)', flush=True)

        all_items = pd.Index(data['item'].unique())
        for seed in args.seeds:
            train, test = user_holdout_split(data, seed=seed, test_frac=0.2)

            # --- leakage / split integrity checks -------------------------
            tr_pairs = set(map(tuple, train[['user', 'item']].values))
            te_pairs = set(map(tuple, test[['user', 'item']].values))
            overlap = len(tr_pairs & te_pairs)

            eval_users = test['user'].drop_duplicates().tolist()
            cand_map = build_candidate_map(train, eval_users, all_items)
            seen = train.groupby('user')['item'].agg(set).to_dict()
            probe = eval_users[:200]
            cand_leak = sum(len(set(cand_map[u]) & seen.get(u, set())) for u in probe)

            print(f'  [{ds_name} seed={seed}] train={len(train):,} test={len(test):,} '
                  f'eval_users={len(eval_users):,} | train/test overlap={overlap} '
                  f'| candidate leakage (first {len(probe)} users)={cand_leak}',
                  flush=True)

            candidate_fn = lambda u, cm=cand_map: cm.get(u, [])

            for alg_name in args.algos:
                kind, factory = ALGOS[alg_name]
                t1 = time.time()
                try:
                    algo = factory()
                    model = algo.fit(train)
                    recs = batch.recommend(model, eval_users, TOPN,
                                           candidates=candidate_fn, n_jobs=1)
                    ranked = ensure_rank(recs)
                    nm = node_metrics(recs, test, ks=KS)
                    node_vals = {}
                    for _, r in nm.iterrows():
                        node_vals[f'Precision@{int(r.k)}'] = r['Precision']
                        node_vals[f'nDCG@{int(r.k)}'] = r['nDCG']
                    ind_vals = independent_metrics(ranked, test, ks=KS)

                    scores = recs['score'].dropna()
                    row = {
                        'dataset': ds_name, 'algorithm': alg_name, 'seed': seed,
                        'kind': kind,
                        'n_recs': len(recs),
                        'distinct_scores': int(scores.nunique()),
                        'score_std': float(scores.std()) if len(scores) else np.nan,
                        'train_test_overlap': overlap,
                        'candidate_leakage_probe': cand_leak,
                        'runtime_s': round(time.time() - t1, 1),
                    }
                    for k in KS:
                        row[f'rerun_P@{k}'] = node_vals[f'Precision@{k}']
                        row[f'rerun_nDCG@{k}'] = node_vals[f'nDCG@{k}']
                        row[f'indep_P@{k}'] = ind_vals[f'Precision@{k}']
                        row[f'indep_nDCG@{k}'] = ind_vals[f'nDCG@{k}']

                    if kind == 'node':
                        m = reported[(reported.dataset == ds_name)
                                     & (reported.algorithm == alg_name)
                                     & (reported.seed == seed)]
                        if len(m) == 1:
                            for k in KS:
                                row[f'reported_P@{k}'] = float(m[f'Precision@{k}'].iloc[0])
                                row[f'reported_nDCG@{k}'] = float(m[f'nDCG@{k}'].iloc[0])

                    append_row(row)
                    rp = row.get('reported_nDCG@10')
                    print(f'    {alg_name:18s} nDCG@10 rerun={row["rerun_nDCG@10"]:.6f} '
                          f'indep={row["indep_nDCG@10"]:.6f} '
                          f'reported={rp if rp is None else f"{rp:.6f}"} '
                          f'distinct_scores={row["distinct_scores"]} '
                          f'({row["runtime_s"]}s)', flush=True)
                except Exception as e:
                    print(f'    {alg_name:18s} FAILED: {type(e).__name__}: {e}',
                          flush=True)
                    append_row({'dataset': ds_name, 'algorithm': alg_name,
                                'seed': seed, 'kind': kind,
                                'error': f'{type(e).__name__}: {e}'})

    print('\nDone. Results in', OUT_CSV)


if __name__ == '__main__':
    main()
