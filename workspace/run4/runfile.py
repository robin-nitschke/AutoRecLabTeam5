import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import warnings
warnings.filterwarnings('ignore')

import inspect
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t

from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem
from lenskit import batch

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'timestamps': []},
    'amazon_videogames': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'timestamps': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'timestamps': []},
}


def robust_read_csv(path, **kwargs):
    kw = dict(kwargs)
    kw.setdefault('engine', 'python')
    kw.setdefault('on_bad_lines', 'skip')
    return pd.read_csv(path, **kw)


def standardize_ui(df, user_col, item_col):
    out = df[[user_col, item_col]].copy()
    out.columns = ['user', 'item']
    out['user'] = out['user'].astype(str).str.strip()
    out['item'] = out['item'].astype(str).str.strip()
    out = out.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    out = out.dropna().drop_duplicates().reset_index(drop=True)
    out['rating'] = 1.0
    return out[['user', 'item', 'rating']]


def k_core_filter(df, user_col='user', item_col='item', min_uc=5, min_ic=5):
    cur = df[[user_col, item_col]].drop_duplicates().copy()
    while True:
        ucnt = cur[user_col].value_counts()
        icnt = cur[item_col].value_counts()
        keep_u = set(ucnt[ucnt >= min_uc].index)
        keep_i = set(icnt[icnt >= min_ic].index)
        nxt = cur[cur[user_col].isin(keep_u) & cur[item_col].isin(keep_i)].copy()
        if len(nxt) == len(cur):
            break
        cur = nxt
    cur['rating'] = 1.0
    return cur.reset_index(drop=True)


def remove_users_with_lt2(df):
    vc = df['user'].value_counts()
    keep = vc[vc >= 2].index
    return df[df['user'].isin(keep)].reset_index(drop=True), int((vc < 2).sum())


def load_ml100k(path='u.data'):
    raw = robust_read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    raw['rating'] = pd.to_numeric(raw['rating'], errors='coerce')
    raw = raw.dropna(subset=['user', 'item', 'rating'])
    pre_n = len(raw)
    raw = raw[raw['rating'] > 3]
    thr_n = len(raw)
    ui = standardize_ui(raw, 'user', 'item')
    dedup_n = len(ui)
    ui, dropped_lt2 = remove_users_with_lt2(ui)
    kc = k_core_filter(ui)
    summary = {
        'dataset': 'ml100k', 'raw_rows': pre_n, 'after_threshold_rows': thr_n, 'after_dedup_rows': dedup_n,
        'dropped_users_lt2_pre_split': dropped_lt2, 'final_interactions': len(kc), 'final_users': kc.user.nunique(), 'final_items': kc.item.nunique()
    }
    return kc, summary


def load_amazon(path='VideoGames.csv'):
    df = None
    for opts in [dict(header=None, names=['user', 'item', 'rating', 'timestamp']), dict(header=0)]:
        try:
            cand = robust_read_csv(path, **opts)
            cols = list(cand.columns)
            if {'user', 'item', 'rating'}.issubset(set(cols)):
                df = cand[['user', 'item', 'rating']].copy()
                break
            if len(cols) >= 3:
                cand = cand.iloc[:, :4].copy()
                cand.columns = ['user', 'item', 'rating', 'timestamp'][:cand.shape[1]]
                df = cand[['user', 'item', 'rating']].copy()
                break
        except Exception:
            pass
    if df is None:
        raise RuntimeError('Could not parse VideoGames.csv')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['user', 'item', 'rating'])
    pre_n = len(df)
    df = df[df['rating'] > 3]
    thr_n = len(df)
    ui = standardize_ui(df, 'user', 'item')
    dedup_n = len(ui)
    ui, dropped_lt2 = remove_users_with_lt2(ui)
    kc = k_core_filter(ui)
    summary = {
        'dataset': 'amazon_videogames', 'raw_rows': pre_n, 'after_threshold_rows': thr_n, 'after_dedup_rows': dedup_n,
        'dropped_users_lt2_pre_split': dropped_lt2, 'final_interactions': len(kc), 'final_users': kc.user.nunique(), 'final_items': kc.item.nunique()
    }
    return kc, summary


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    df = robust_read_csv(path, sep='\t')
    lower = {c.lower(): c for c in df.columns}
    user_col = lower.get('userid', df.columns[0])
    item_col = lower.get('artistid', df.columns[1])
    df = df.dropna(subset=[user_col, item_col])
    pre_n = len(df)
    ui = standardize_ui(df, user_col, item_col)
    dedup_n = len(ui)
    ui, dropped_lt2 = remove_users_with_lt2(ui)
    kc = k_core_filter(ui)
    summary = {
        'dataset': 'lastfm', 'raw_rows': pre_n, 'after_threshold_rows': pre_n, 'after_dedup_rows': dedup_n,
        'dropped_users_lt2_pre_split': dropped_lt2, 'final_interactions': len(kc), 'final_users': kc.user.nunique(), 'final_items': kc.item.nunique()
    }
    return kc, summary


def user_holdout_split(df, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    skipped = 0
    for _, g in df.groupby('user', sort=False):
        n = len(g)
        if n < 2:
            skipped += 1
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n - 1, n_test)
        idx = np.arange(n)
        rng.shuffle(idx)
        mask = np.zeros(n, dtype=bool)
        mask[idx[:n_test]] = True
        train_parts.append(g.iloc[~mask])
        test_parts.append(g.iloc[mask])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    train['rating'] = 1.0
    test['rating'] = 1.0
    return train[['user', 'item', 'rating']], test[['user', 'item', 'rating']], skipped


def make_als(seed):
    sig = inspect.signature(ImplicitMF)
    kwargs = {}
    if 'features' in sig.parameters:
        kwargs['features'] = 50
    if 'iterations' in sig.parameters:
        kwargs['iterations'] = 15
    if 'reg' in sig.parameters:
        kwargs['reg'] = 0.1
    if 'weight' in sig.parameters:
        kwargs['weight'] = 40
    if 'method' in sig.parameters:
        kwargs['method'] = 'cg'
    if 'rng_spec' in sig.parameters:
        kwargs['rng_spec'] = seed
    elif 'random_state' in sig.parameters:
        kwargs['random_state'] = seed
    return Recommender.adapt(ImplicitMF(**kwargs))


def make_itemknn():
    for kwargs in ({'nnbrs': 20, 'feedback': 'implicit'}, {'nnbrs': 20}, {'k': 20}, {}):
        try:
            return Recommender.adapt(ItemItem(**kwargs))
        except Exception:
            pass
    return Recommender.adapt(ItemItem(20))


def get_algorithms(seed):
    return {'ALS': make_als(seed), 'ItemKNN': make_itemknn(), 'Pop': Recommender.adapt(Popular())}


def precision_at_k(rec_items, truth, k):
    return sum(1 for x in rec_items[:k] if x in truth) / k if k else 0.0


def ndcg_at_k(rec_items, truth, k):
    rec_k = rec_items[:k]
    dcg = sum((1.0 / np.log2(i + 2)) for i, x in enumerate(rec_k) if x in truth)
    ideal = min(len(truth), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg


def recommend_excluding_seen(model, users, train, n=10):
    seen = train.groupby('user')['item'].apply(set).to_dict()
    out = []
    for u in users:
        extra = len(seen.get(u, set()))
        try:
            recs = batch.recommend(model, [u], n + extra)
        except Exception:
            recs = batch.recommend(model, [u], n)
        if recs is None or len(recs) == 0 or 'item' not in recs.columns:
            continue
        recs = recs[recs['user'] == u].copy()
        recs['item'] = recs['item'].astype(str)
        recs = recs[~recs['item'].isin(set(map(str, seen.get(u, set()))))].head(n)
        recs['user'] = u
        keep = ['user', 'item'] + [c for c in ['rank', 'score'] if c in recs.columns]
        out.append(recs[keep])
    cols = ['user', 'item', 'rank', 'score']
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def evaluate_model(model, train, test, algo_name, dataset_name, seed):
    model.fit(train)
    users = sorted(test['user'].unique())
    truth = test.groupby('user')['item'].apply(lambda x: set(map(str, x))).to_dict()
    recs = recommend_excluding_seen(model, users, train, n=max(KS))
    user_recs = recs.groupby('user')['item'].apply(list).to_dict() if len(recs) else {}
    rows = []
    for u in users:
        r, tset = user_recs.get(u, []), truth.get(u, set())
        row = {'dataset': dataset_name, 'algorithm': algo_name, 'seed': seed, 'user': u}
        for k in KS:
            row[f'P@{k}'] = precision_at_k(r, tset, k)
            row[f'nDCG@{k}'] = ndcg_at_k(r, tset, k)
        rows.append(row)
    udf = pd.DataFrame(rows)
    metric_cols = [f'P@{k}' for k in KS] + [f'nDCG@{k}' for k in KS]
    summary = udf[metric_cols].mean().to_dict()
    val_loss = 1 - summary['nDCG@10']
    print(f'Epoch {seed}: validation_loss = {val_loss:.4f} | {dataset_name} | {algo_name}')
    print({k: round(v, 4) for k, v in summary.items()})
    experiment_data[dataset_name]['metrics']['val'].append({'timestamp': seed, 'seed': seed, 'algorithm': algo_name, **summary})
    experiment_data[dataset_name]['losses']['val'].append({'timestamp': seed, 'seed': seed, 'algorithm': algo_name, 'validation_loss': val_loss})
    experiment_data[dataset_name]['predictions'].append(recs[['user', 'item']].astype(str).to_dict('records'))
    experiment_data[dataset_name]['ground_truth'].append(test[['user', 'item']].astype(str).to_dict('records'))
    experiment_data[dataset_name]['timestamps'].append(seed)
    return udf


loaded = {
    'ml100k': load_ml100k('u.data'),
    'amazon_videogames': load_amazon('VideoGames.csv'),
    'lastfm': load_lastfm('UserTaggedArtists-timestamps.dat'),
}

datasets = {k: v[0] for k, v in loaded.items()}
prep_df = pd.DataFrame([v[1] for v in loaded.values()])
print(prep_df.to_string(index=False))
prep_df.to_csv(os.path.join(working_dir, 'preprocessing_summary.csv'), index=False)
np.save(os.path.join(working_dir, 'preprocessing_summary.npy'), prep_df.to_dict('records'), allow_pickle=True)

all_user_metrics, all_seed_metrics = [], []
for dname, df in datasets.items():
    print(f'Loaded {dname}: {len(df):,} interactions, {df.user.nunique():,} users, {df.item.nunique():,} items')
    for seed in SEEDS:
        train, test, skipped = user_holdout_split(df, test_ratio=0.2, seed=seed)
        experiment_data[dname]['metrics']['train'].append({'timestamp': seed, 'seed': seed, 'n_train': len(train), 'n_test': len(test), 'skipped_users': skipped})
        algos = get_algorithms(seed)
        for aname, algo in algos.items():
            udf = evaluate_model(algo, train, test, aname, dname, seed)
            all_user_metrics.append(udf)
            means = udf[[f'P@{k}' for k in KS] + [f'nDCG@{k}' for k in KS]].mean().to_dict()
            means.update({'dataset': dname, 'algorithm': aname, 'seed': seed, 'n_users_eval': udf.user.nunique()})
            all_seed_metrics.append(means)

user_metrics_df = pd.concat(all_user_metrics, ignore_index=True)
seed_metrics_df = pd.DataFrame(all_seed_metrics)

summary_rows = []
for (d, a), g in seed_metrics_df.groupby(['dataset', 'algorithm']):
    row = {'dataset': d, 'algorithm': a}
    for m in [f'P@{k}' for k in KS] + [f'nDCG@{k}' for k in KS]:
        vals = g[m].to_numpy()
        row[f'{m}_mean'] = float(np.mean(vals))
        row[f'{m}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row[f'{m}_min'] = float(np.min(vals))
        row[f'{m}_max'] = float(np.max(vals))
        row[f'{m}_range'] = float(np.max(vals) - np.min(vals))
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)

stats_rows = []
for dname in datasets:
    sub = seed_metrics_df[seed_metrics_df['dataset'] == dname]
    for metric in ['nDCG@10', 'P@10']:
        base = sub[sub['algorithm'] == 'Pop'].sort_values('seed')
        for algo in ['ALS', 'ItemKNN']:
            comp = sub[sub['algorithm'] == algo].sort_values('seed')
            merged = base[['seed', metric]].merge(comp[['seed', metric]], on='seed', suffixes=('_pop', f'_{algo}'))
            if len(merged) >= 2:
                diff = merged[f'{metric}_{algo}'] - merged[f'{metric}_pop']
                stat, pval = ttest_rel(merged[f'{metric}_{algo}'], merged[f'{metric}_pop'])
                mean_diff = float(diff.mean())
                sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
                se = sd / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
                ci = float(t.ppf(0.975, df=len(diff)-1) * se) if len(diff) > 1 else 0.0
                stats_rows.append({'dataset': dname, 'comparison': f'{algo} vs Pop', 'metric': metric, 'mean_diff': mean_diff, 't_stat': float(stat), 'p_value': float(pval), 'ci95_low': mean_diff - ci, 'ci95_high': mean_diff + ci})
stats_df = pd.DataFrame(stats_rows)

print('\nAggregate results over seeds (mean/std/range):')
print(summary_df.round(4).to_string(index=False))
print('\nPaired t-tests across seeds:')
print(stats_df.round(4).to_string(index=False) if len(stats_df) else 'No valid paired tests.')

seed_metrics_df.to_csv(os.path.join(working_dir, 'seed_metrics.csv'), index=False)
user_metrics_df.to_csv(os.path.join(working_dir, 'user_metrics.csv'), index=False)
summary_df.to_csv(os.path.join(working_dir, 'aggregate_summary.csv'), index=False)
stats_df.to_csv(os.path.join(working_dir, 'stats_tests.csv'), index=False)

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'seed_metrics.npy'), seed_metrics_df.to_dict('records'), allow_pickle=True)
np.save(os.path.join(working_dir, 'user_metrics.npy'), user_metrics_df.to_dict('records'), allow_pickle=True)
np.save(os.path.join(working_dir, 'aggregate_summary.npy'), summary_df.to_dict('records'), allow_pickle=True)
np.save(os.path.join(working_dir, 'stats_tests.npy'), stats_df.to_dict('records'), allow_pickle=True)
np.savez_compressed(
    os.path.join(working_dir, 'all_outputs.npz'),
    seed_metrics=seed_metrics_df.to_records(index=False),
    user_metrics=user_metrics_df.to_records(index=False),
    aggregate_summary=summary_df.to_records(index=False),
    stats_tests=stats_df.to_records(index=False),
    preprocessing=prep_df.to_records(index=False)
)

print(f'Artifacts saved to: {working_dir}')