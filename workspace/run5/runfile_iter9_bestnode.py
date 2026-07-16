import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from lenskit import batch
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'amazon_videogames': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
}

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]
TOPN = max(KS)


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
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
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
        df = pd.read_csv(path, header=None, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2])
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
    test = test[test['user'].isin(train_users) & test['item'].isin(train_items)].reset_index(drop=True)
    return train.reset_index(drop=True), test


def make_algorithms():
    return {
        'ALS': ImplicitMF(features=50, iterations=15, weight=40, method='cg', use_ratings=False),
        'ItemKNN': ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False),
        'Pop': Popular(),
    }


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


def precision_ndcg_at_k(recs, test, ks=(1, 5, 10)):
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
        idcg = pd.Series({u: np.sum(1 / np.log2(np.arange(2, min(len(truth[u]), k) + 2))) for u in users})
        ndcg_u = (dcg_u / idcg.replace(0, np.nan)).fillna(0)
        rows.append({'k': k, 'Precision': float(prec_u.mean()), 'nDCG': float(ndcg_u.mean())})
    return pd.DataFrame(rows)


def paired_stats(df, metric_col='value'):
    out = []
    for ds in df['dataset'].unique():
        sub = df[df['dataset'] == ds]
        piv = sub.pivot_table(index='seed', columns='algorithm', values=metric_col)
        algs = list(piv.columns)
        for i in range(len(algs)):
            for j in range(i + 1, len(algs)):
                a, b = algs[i], algs[j]
                pair = piv[[a, b]].dropna()
                d = pair[a] - pair[b]
                if len(pair) >= 2:
                    t, p = stats.ttest_rel(pair[a], pair[b], nan_policy='omit')
                else:
                    t, p = np.nan, np.nan
                out.append({'dataset': ds, 'alg_a': a, 'alg_b': b, 'mean_diff': float(d.mean()) if len(d) else np.nan, 't_stat': float(t) if pd.notna(t) else np.nan, 'p_value': float(p) if pd.notna(p) else np.nan})
    return pd.DataFrame(out)


def plot_metric_variation(results, metric_name):
    for ds in results['dataset'].unique():
        sub = results[results['dataset'] == ds]
        plt.figure(figsize=(7, 4))
        for alg in sub['algorithm'].unique():
            s = sub[sub['algorithm'] == alg].sort_values('seed')
            plt.plot(s['seed'], s[metric_name], marker='o', label=alg)
        plt.title(f'{ds}: {metric_name} across split seeds')
        plt.xlabel('Seed')
        plt.ylabel(metric_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f'{ds}_{metric_name.replace("@", "at")}_seed_variation.png'), dpi=150)
        plt.close()


datasets = {
    'ml100k': load_ml100k('u.data'),
    'amazon_videogames': load_amazon('VideoGames.csv'),
    'lastfm': load_lastfm('UserTaggedArtists-timestamps.dat'),
}

all_results = []

for ds_name, data in datasets.items():
    print(f'Loaded {ds_name}: {len(data):,} interactions, {data.user.nunique():,} users, {data.item.nunique():,} items')
    all_items = pd.Index(data['item'].unique())
    for seed in SEEDS:
        train, test = user_holdout_split(data, seed=seed, test_frac=0.2)
        eval_users = test['user'].drop_duplicates().tolist()
        cand_map = build_candidate_map(train, eval_users, all_items)
        candidate_fn = lambda u, cm=cand_map: cm.get(u, [])
        algos = make_algorithms()
        for alg_name, algo in algos.items():
            print(f'Epoch {seed}: validation_loss = nan')
            model = algo.fit(train)
            recs = batch.recommend(model, eval_users, TOPN, candidates=candidate_fn)
            recs = ensure_rank(recs)
            mets = precision_ndcg_at_k(recs[['user', 'item', 'rank']].copy(), test, ks=KS)
            row = {'dataset': ds_name, 'algorithm': alg_name, 'seed': seed}
            for _, r in mets.iterrows():
                row[f'Precision@{int(r.k)}'] = r['Precision']
                row[f'nDCG@{int(r.k)}'] = r['nDCG']
            all_results.append(row)
            experiment_data[ds_name]['metrics']['val'].append({'seed': seed, 'algorithm': alg_name, 'metrics': row})
            experiment_data[ds_name]['losses']['val'].append({'seed': seed, 'algorithm': alg_name, 'validation_loss': np.nan})
            experiment_data[ds_name]['predictions'].append({'seed': seed, 'algorithm': alg_name, 'topn_users': len(eval_users), 'num_recs': len(recs)})
            experiment_data[ds_name]['ground_truth'].append({'seed': seed, 'num_test': len(test)})

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_results.csv'), index=False)

long_rows = []
metric_cols = [c for c in results_df.columns if '@' in c]
for _, r in results_df.iterrows():
    for metric in metric_cols:
        long_rows.append({'dataset': r['dataset'], 'algorithm': r['algorithm'], 'seed': r['seed'], 'metric': metric, 'value': r[metric]})
long_df = pd.DataFrame(long_rows)
long_df.to_csv(os.path.join(working_dir, 'seed_sensitivity_results_long.csv'), index=False)

summary = long_df.groupby(['dataset', 'algorithm', 'metric'])['value'].agg(['mean', 'std']).reset_index()
summary['cv'] = summary['std'] / summary['mean'].replace(0, np.nan)
summary.to_csv(os.path.join(working_dir, 'seed_sensitivity_summary.csv'), index=False)

seed_effect = long_df.groupby(['dataset', 'algorithm', 'metric'])['value'].agg(['min', 'max', 'std']).reset_index()
seed_effect['range'] = seed_effect['max'] - seed_effect['min']
seed_effect.to_csv(os.path.join(working_dir, 'seed_effect_summary.csv'), index=False)

stats_tables = []
for metric in ['Precision@10', 'nDCG@10']:
    st = paired_stats(long_df[long_df['metric'] == metric].copy(), metric_col='value')
    st['metric'] = metric
    stats_tables.append(st)
stats_df = pd.concat(stats_tables, ignore_index=True)
stats_df.to_csv(os.path.join(working_dir, 'paired_stats.csv'), index=False)

plot_metric_variation(results_df, 'Precision@10')
plot_metric_variation(results_df, 'nDCG@10')

for ds_name in experiment_data:
    np.save(os.path.join(working_dir, f'{ds_name}_metrics.npy'), np.array(experiment_data[ds_name]['metrics']['val'], dtype=object))
    np.save(os.path.join(working_dir, f'{ds_name}_losses.npy'), np.array(experiment_data[ds_name]['losses']['val'], dtype=object))
    np.save(os.path.join(working_dir, f'{ds_name}_predictions.npy'), np.array(experiment_data[ds_name]['predictions'], dtype=object))
    np.save(os.path.join(working_dir, f'{ds_name}_ground_truth.npy'), np.array(experiment_data[ds_name]['ground_truth'], dtype=object))
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.savez_compressed(os.path.join(working_dir, 'results_arrays.npz'), results=results_df.to_records(index=False), summary=summary.to_records(index=False), stats=stats_df.to_records(index=False), seed_effect=seed_effect.to_records(index=False))

print('\nPer-seed results:')
print(results_df.to_string(index=False))
print('\nAcross-seed summary (mean/std/cv):')
print(summary.to_string(index=False))
print('\nSeed effect summary (range/std):')
print(seed_effect.to_string(index=False))
print('\nPaired t-tests on seed-matched runs (Precision@10 and nDCG@10):')
print(stats_df.to_string(index=False))
