import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

from lenskit import batch
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]
TOPN = max(KS)

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'amazon_videogames': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
}


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Required file not found: {path}')


def finalize_interactions(df, name):
    if df is None or len(df) == 0:
        raise ValueError(f'{name}: no data loaded')
    if not {'user', 'item'}.issubset(df.columns):
        raise ValueError(f"{name}: required columns ['user','item'] not found")
    df = df[['user', 'item']].dropna().drop_duplicates().copy()
    if len(df) == 0:
        raise ValueError(f'{name}: empty after dropping missing/duplicate interactions')
    return df


def k_core_filter(df, min_uc=5, min_ic=5):
    df = finalize_interactions(df, 'k_core_input')
    changed = True
    while changed and len(df) > 0:
        n0 = len(df)
        uc = df['user'].value_counts()
        ic = df['item'].value_counts()
        df = df[df['user'].isin(uc[uc >= min_uc].index)]
        df = df[df['item'].isin(ic[ic >= min_ic].index)]
        changed = len(df) != n0
    if len(df) == 0:
        raise ValueError('Dataset empty after 5-core filtering')
    return df.reset_index(drop=True)


def load_ml100k(path='u.data'):
    require_file(path)
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    if not {'user', 'item', 'rating'}.issubset(df.columns):
        raise ValueError('ml100k: invalid schema')
    df = df[df['rating'] > 3][['user', 'item']]
    return k_core_filter(df)


def load_amazon(path='VideoGames.csv'):
    require_file(path)
    df0 = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df0.columns}
    explicit_sets = [
        ('userid', 'productid', 'score'),
        ('user_id', 'item_id', 'rating'),
        ('user', 'item', 'rating'),
        ('reviewerid', 'asin', 'overall'),
    ]
    found = None
    for u, i, r in explicit_sets:
        if u in cols and i in cols and r in cols:
            found = (cols[u], cols[i], cols[r])
            break
    if found is not None:
        df = df0.loc[:, list(found)].copy()
        df.columns = ['user', 'item', 'rating']
    else:
        df = pd.read_csv(path, header=None, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2])
    if not {'user', 'item', 'rating'}.issubset(df.columns):
        raise ValueError('amazon_videogames: invalid schema')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'] > 3][['user', 'item']]
    return k_core_filter(df)


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    require_file(path)
    df = pd.read_csv(path, sep='\t')
    low = {str(c).strip().lower(): c for c in df.columns}
    user_col = low.get('userid')
    item_col = low.get('artistid')
    if user_col is None or item_col is None:
        raise ValueError('lastfm: expected userid and artistid columns')
    df = df[[user_col, item_col]].copy()
    df.columns = ['user', 'item']
    return k_core_filter(df)


def user_holdout_split(df, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, g in df.groupby('user', sort=False):
        n = len(g)
        if n < 2:
            continue
        idx = np.arange(n)
        n_test = max(1, int(np.floor(n * test_frac)))
        n_test = min(n_test, n - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True
        train_parts.append(g.iloc[~mask])
        test_parts.append(g.iloc[mask])
    if not train_parts or not test_parts:
        raise ValueError('Split failed: no train/test parts created')
    train = pd.concat(train_parts, ignore_index=True).drop_duplicates()
    test = pd.concat(test_parts, ignore_index=True).drop_duplicates()
    if len(train) == 0 or len(test) == 0:
        raise ValueError('Split failed: empty train or test')
    train_users = set(train['user'].unique())
    train_items = set(train['item'].unique())
    test = test[test['user'].isin(train_users) & test['item'].isin(train_items)].reset_index(drop=True)
    tu = test.groupby('user').size()
    valid_users = tu[tu > 0].index
    test = test[test['user'].isin(valid_users)].reset_index(drop=True)
    train = train[train['user'].isin(valid_users)].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0 or test['user'].nunique() == 0:
        raise ValueError('Split failed after filtering invalid test users/items')
    return train, test


def make_algorithms():
    return {
        'ALS': ImplicitMF(),
        'ItemKNN': ItemItem(),
        'Pop': Popular(),
    }


def build_candidate_map(train, users, all_items):
    all_items = set(all_items)
    seen = train.groupby('user')['item'].agg(set).to_dict()
    return {u: list(all_items - seen.get(u, set())) for u in users}


def ensure_rank(recs):
    recs = recs.copy()
    if len(recs) == 0:
        return pd.DataFrame(columns=['user', 'item', 'rank'])
    if 'rank' not in recs.columns:
        sort_cols = ['user'] + (['score'] if 'score' in recs.columns else ['item'])
        asc = [True, False] if 'score' in recs.columns else [True, True]
        recs = recs.sort_values(sort_cols, ascending=asc)
        recs['rank'] = recs.groupby('user').cumcount() + 1
    return recs[['user', 'item', 'rank']].reset_index(drop=True)


def precision_ndcg_at_k(recs, test, ks=(1, 5, 10)):
    truth = test.groupby('user')['item'].agg(set).to_dict()
    users = pd.Index(sorted(truth.keys(), key=lambda x: str(x)), name='user')
    if len(users) == 0:
        raise ValueError('No evaluation users')
    recs = ensure_rank(recs)
    recs = recs[recs['user'].isin(users) & recs['rank'].isin(range(1, max(ks) + 1))].copy()
    gt = test[['user', 'item']].drop_duplicates().copy()
    gt['rel'] = 1
    merged = recs.merge(gt, on=['user', 'item'], how='left')
    merged['rel'] = merged['rel'].fillna(0).astype(int)
    merged['discount'] = 1.0 / np.log2(merged['rank'] + 1)
    merged['gain'] = merged['rel'] * merged['discount']
    rows = []
    truth_sizes = pd.Series({u: len(truth[u]) for u in users})
    for k in ks:
        mk = merged[merged['rank'] <= k]
        hits = mk.groupby('user')['rel'].sum().reindex(users, fill_value=0)
        dcg = mk.groupby('user')['gain'].sum().reindex(users, fill_value=0.0)
        idcg = truth_sizes.apply(lambda n: np.sum(1.0 / np.log2(np.arange(2, min(n, k) + 2))))
        prec = (hits / k).mean()
        ndcg = (dcg / idcg.replace(0, np.nan)).fillna(0.0).mean()
        rows.append({'metric': 'Precision', 'k': k, 'value': float(prec)})
        rows.append({'metric': 'nDCG', 'k': k, 'value': float(ndcg)})
    return pd.DataFrame(rows)


def paired_stats(long_df):
    out = []
    for (ds, metric, k), sub in long_df.groupby(['dataset', 'metric', 'k']):
        piv = sub.pivot_table(index='seed', columns='algorithm', values='value')
        algs = list(piv.columns)
        for i in range(len(algs)):
            for j in range(i + 1, len(algs)):
                a, b = algs[i], algs[j]
                pair = piv[[a, b]].dropna()
                if len(pair) >= 2:
                    t, p = stats.ttest_rel(pair[a], pair[b], nan_policy='omit')
                    diff = pair[a] - pair[b]
                    out.append({'dataset': ds, 'metric': metric, 'k': k, 'alg_a': a, 'alg_b': b, 'n_seeds': int(len(pair)), 'mean_diff': float(diff.mean()), 't_stat': float(t), 'p_value': float(p), 'note': 'paired t-test on 5 seed-matched runs; descriptive only'})
    return pd.DataFrame(out)


datasets = {
    'ml100k': load_ml100k('u.data'),
    'amazon_videogames': load_amazon('VideoGames.csv'),
    'lastfm': load_lastfm('UserTaggedArtists-timestamps.dat'),
}

results_long = []
results_wide = []

for ds_name, data in datasets.items():
    data = finalize_interactions(data, ds_name)
    if len(data) == 0:
        raise ValueError(f'{ds_name}: empty dataset after preprocessing')
    print(f'Loaded {ds_name}: {len(data):,} interactions, {data.user.nunique():,} users, {data.item.nunique():,} items')
    all_items = pd.Index(data['item'].unique())
    for seed in SEEDS:
        train, test = user_holdout_split(data, seed=seed, test_frac=0.2)
        eval_users = test['user'].drop_duplicates().tolist()
        cand_map = build_candidate_map(train, eval_users, all_items)
        candidate_fn = lambda u, cm=cand_map: cm.get(u, [])
        row = {'dataset': ds_name, 'seed': seed}
        for alg_name, algo in make_algorithms().items():
            print(f'Epoch {seed}: validation_loss = nan')
            try:
                model = algo.fit(train)
                recs = batch.recommend(model, eval_users, TOPN, candidates=candidate_fn)
                recs = ensure_rank(recs)
                mets = precision_ndcg_at_k(recs, test, ks=KS)
            except Exception as e:
                print(f'{ds_name} seed={seed} alg={alg_name} failed: {e}')
                mets = pd.DataFrame([{'metric': m, 'k': k, 'value': np.nan} for m in ['Precision', 'nDCG'] for k in KS])
                recs = pd.DataFrame(columns=['user', 'item', 'rank'])
            for _, mr in mets.iterrows():
                results_long.append({'dataset': ds_name, 'algorithm': alg_name, 'seed': seed, 'metric': mr['metric'], 'k': int(mr['k']), 'value': float(mr['value']) if pd.notna(mr['value']) else np.nan})
                row[f"{alg_name}_{mr['metric']}@{int(mr['k'])}"] = mr['value']
            experiment_data[ds_name]['metrics']['val'].append({'seed': seed, 'algorithm': alg_name, 'timestamp': pd.Timestamp.utcnow().isoformat(), 'metrics': mets.to_dict('records')})
            experiment_data[ds_name]['losses']['val'].append({'seed': seed, 'algorithm': alg_name, 'timestamp': pd.Timestamp.utcnow().isoformat(), 'validation_loss': np.nan})
            experiment_data[ds_name]['predictions'].append({'seed': seed, 'algorithm': alg_name, 'timestamp': pd.Timestamp.utcnow().isoformat(), 'num_recs': int(len(recs))})
            experiment_data[ds_name]['ground_truth'].append({'seed': seed, 'algorithm': alg_name, 'timestamp': pd.Timestamp.utcnow().isoformat(), 'num_test': int(len(test)), 'num_eval_users': int(len(eval_users))})
        results_wide.append(row)

results_df = pd.DataFrame(results_wide)
long_df = pd.DataFrame(results_long)
summary = long_df.groupby(['dataset', 'algorithm', 'metric', 'k'])['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
summary['range'] = summary['max'] - summary['min']
summary['cv'] = summary['std'] / summary['mean'].replace(0, np.nan)
stats_df = paired_stats(long_df)

results_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_results_wide.csv'), index=False)
long_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_results_long_tidy.csv'), index=False)
summary.to_csv(os.path.join(working_dir, 'seed_sensitivity_summary.csv'), index=False)
stats_df.to_csv(os.path.join(working_dir, 'paired_stats_descriptive.csv'), index=False)

for ds_name in experiment_data:
    np.save(os.path.join(working_dir, f'{ds_name}_metrics.npy'), np.array(experiment_data[ds_name]['metrics']['val'], dtype=object))
    np.save(os.path.join(working_dir, f'{ds_name}_losses.npy'), np.array(experiment_data[ds_name]['losses']['val'], dtype=object))
    np.save(os.path.join(working_dir, f'{ds_name}_predictions.npy'), np.array(experiment_data[ds_name]['predictions'], dtype=object))
    np.save(os.path.join(working_dir, f'{ds_name}_ground_truth.npy'), np.array(experiment_data[ds_name]['ground_truth'], dtype=object))
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.savez_compressed(
    os.path.join(working_dir, 'results_arrays.npz'),
    results_wide=results_df.to_records(index=False),
    results_long=long_df.to_records(index=False),
    summary=summary.to_records(index=False),
    stats=stats_df.to_records(index=False) if len(stats_df) else np.array([], dtype=object),
)

print('\nTidy per-run results:')
print(long_df.sort_values(['dataset', 'algorithm', 'seed', 'metric', 'k']).to_string(index=False))
print('\nAcross-seed summary:')
print(summary.sort_values(['dataset', 'metric', 'k', 'algorithm']).to_string(index=False))
print('\nPaired statistical comparison (descriptive, interpret cautiously with 5 seeds):')
print(stats_df.sort_values(['dataset', 'metric', 'k', 'alg_a', 'alg_b']).to_string(index=False) if len(stats_df) else 'No paired stats available')