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
from lenskit.algorithms import Recommender
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
MAX_K = max(KS)


def k_core_filter(df, user_col='user', item_col='item', min_k=5):
    df = df[[user_col, item_col]].drop_duplicates().copy()
    while True:
        uc = df[user_col].value_counts()
        ic = df[item_col].value_counts()
        good_u = uc[uc >= min_k].index
        good_i = ic[ic >= min_k].index
        new_df = df[df[user_col].isin(good_u) & df[item_col].isin(good_i)]
        if len(new_df) == len(df):
            break
        df = new_df
    return df.reset_index(drop=True)


def load_ml100k(path='u.data'):
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] > 3][['user', 'item']].copy()
    return k_core_filter(df)


def load_amazon(path='VideoGames.csv'):
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    ucol = cols.get('user_id', cols.get('userid', cols.get('reviewerid', raw.columns[0])))
    icol = cols.get('item_id', cols.get('asin', raw.columns[1]))
    rcol = cols.get('rating', cols.get('overall', raw.columns[2] if len(raw.columns) > 2 else raw.columns[-1]))
    df = raw[[ucol, icol, rcol]].rename(columns={ucol: 'user', icol: 'item', rcol: 'rating'})
    df = df[df['rating'] > 3][['user', 'item']].copy()
    return k_core_filter(df)


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    df = pd.read_csv(path, sep='\t')
    cols = {c.lower(): c for c in df.columns}
    ucol = cols.get('userid', df.columns[0])
    icol = cols.get('artistid', df.columns[1])
    out = df[[ucol, icol]].rename(columns={ucol: 'user', icol: 'item'})
    return k_core_filter(out)


def user_holdout_split(df, seed=42, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, udf in df.groupby('user', sort=False):
        n = len(udf)
        n_test = max(1, int(np.floor(n * test_frac)))
        if n - n_test < 1:
            n_test = n - 1
        idx = np.arange(n)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True
        test_parts.append(udf.iloc[mask])
        train_parts.append(udf.iloc[~mask])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def build_algorithms():
    return {
        'ALS': Recommender.adapt(ImplicitMF(features=50, iterations=15, reg=0.01, weight=40, method='cg')),
        'ItemKNN': Recommender.adapt(ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False)),
        'Pop': Recommender.adapt(Popular())
    }


def precision_at_k(recs, truth, k):
    vals = []
    for u, items in truth.items():
        ranked = recs.get(u, [])[:k]
        vals.append(sum(1 for it in ranked if it in items) / k)
    return float(np.mean(vals)) if vals else np.nan


def ndcg_at_k(recs, truth, k):
    vals = []
    for u, items in truth.items():
        ranked = recs.get(u, [])[:k]
        dcg = 0.0
        for i, it in enumerate(ranked, start=1):
            if it in items:
                dcg += 1.0 / np.log2(i + 1)
        ideal_hits = min(len(items), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        vals.append((dcg / idcg) if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else np.nan


def evaluate_algo(algo, train, test, ks):
    algo.fit(train)
    users = test['user'].drop_duplicates().tolist()
    try:
        recs = batch.recommend(algo, users, MAX_K, candidates=None)
    except TypeError:
        recs = batch.recommend(algo, users, MAX_K)
    truth = test.groupby('user')['item'].apply(set).to_dict()
    train_items = train.groupby('user')['item'].apply(set).to_dict()
    rec_dict = {}
    for u, udf in recs.groupby('user', sort=False):
        seen = train_items.get(u, set())
        items = [it for it in udf['item'].tolist() if it not in seen][:MAX_K]
        rec_dict[u] = items
    metrics = {}
    for k in ks:
        metrics[f'Precision@{k}'] = precision_at_k(rec_dict, truth, k)
        metrics[f'nDCG@{k}'] = ndcg_at_k(rec_dict, truth, k)
    filtered_rows = []
    for u, items in rec_dict.items():
        for r, it in enumerate(items, start=1):
            filtered_rows.append((u, it, r))
    recs_out = pd.DataFrame(filtered_rows, columns=['user', 'item', 'rank'])
    return metrics, recs_out, truth


datasets = {
    'ml100k': load_ml100k('u.data'),
    'amazon_videogames': load_amazon('VideoGames.csv'),
    'lastfm': load_lastfm('UserTaggedArtists-timestamps.dat')
}

all_results = []
for dname, df in datasets.items():
    print(f'Loaded {dname}: {len(df)} interactions, {df.user.nunique()} users, {df.item.nunique()} items')
    for seed in SEEDS:
        train, test = user_holdout_split(df, seed=seed, test_frac=0.2)
        algos = build_algorithms()
        for aname, algo in algos.items():
            print(f'Epoch {seed}: validation_loss = {float("nan"):.4f}')
            metrics, recs, truth = evaluate_algo(algo, train, test, KS)
            row = {'dataset': dname, 'algorithm': aname, 'seed': seed, 'timestamp': pd.Timestamp.now().isoformat()}
            row.update(metrics)
            all_results.append(row)
            experiment_data[dname]['metrics']['train'].append({'seed': seed, 'algorithm': aname, 'n_train': len(train), 'timestamp': row['timestamp']})
            experiment_data[dname]['metrics']['val'].append(row.copy())
            experiment_data[dname]['losses']['train'].append({'seed': seed, 'algorithm': aname, 'loss': np.nan, 'timestamp': row['timestamp']})
            experiment_data[dname]['losses']['val'].append({'seed': seed, 'algorithm': aname, 'loss': np.nan, 'timestamp': row['timestamp']})
            experiment_data[dname]['predictions'].append({'seed': seed, 'algorithm': aname, 'rows': recs.to_dict('records')})
            experiment_data[dname]['ground_truth'].append({'seed': seed, 'algorithm': aname, 'truth': {str(k): list(v) for k, v in truth.items()}})
            print(dname, aname, seed, {k: round(v, 4) for k, v in metrics.items()})

results = pd.DataFrame(all_results)
results.to_csv(os.path.join(working_dir, 'seed_sensitivity_results.csv'), index=False)
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'results_array.npy'), results.to_records(index=False), allow_pickle=True)

metric_cols = [f'Precision@{k}' for k in KS] + [f'nDCG@{k}' for k in KS]
summary = results.groupby(['dataset', 'algorithm'])[metric_cols].agg(['mean', 'std'])
print('\nMean/std across seeds:')
print(summary)
summary.to_csv(os.path.join(working_dir, 'metric_summary_mean_std.csv'))

cv_rows = []
for (d, a), g in results.groupby(['dataset', 'algorithm']):
    row = {'dataset': d, 'algorithm': a}
    for m in metric_cols:
        mu = g[m].mean()
        sd = g[m].std(ddof=1)
        row[f'{m}_mean'] = mu
        row[f'{m}_std'] = sd
        row[f'{m}_cv'] = sd / mu if pd.notna(mu) and mu != 0 else np.nan
    cv_rows.append(row)
cv_df = pd.DataFrame(cv_rows)
print('\nCoefficient of variation across seeds:')
print(cv_df)
cv_df.to_csv(os.path.join(working_dir, 'cv_summary.csv'), index=False)

stat_rows = []
for d, dg in results.groupby('dataset'):
    for metric in metric_cols:
        piv = dg.pivot(index='seed', columns='algorithm', values=metric)
        algs = [c for c in piv.columns if piv[c].notna().any()]
        for i in range(len(algs)):
            for j in range(i + 1, len(algs)):
                a1, a2 = algs[i], algs[j]
                pair = piv[[a1, a2]].dropna()
                if len(pair) >= 2:
                    t, p = stats.ttest_rel(pair[a1], pair[a2], nan_policy='omit')
                else:
                    t, p = np.nan, np.nan
                stat_rows.append({'dataset': d, 'metric': metric, 'alg1': a1, 'alg2': a2, 'n_pairs': len(pair), 't_stat': t, 'p_value': p})
stat_df = pd.DataFrame(stat_rows)
print('\nPaired t-tests across seeds:')
print(stat_df)
stat_df.to_csv(os.path.join(working_dir, 'paired_ttests_all_metrics.csv'), index=False)

for d in results['dataset'].unique():
    sub = results[results['dataset'] == d]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['nDCG@10', 'Precision@10']):
        grp = sub.groupby('algorithm')[metric].agg(['mean', 'std']).reset_index()
        ax.bar(grp['algorithm'], grp['mean'], yerr=grp['std'].fillna(0), capsize=4)
        ax.set_title(f'{d} {metric}')
        ax.set_ylabel(metric)
    plt.tight_layout()
    fig.savefig(os.path.join(working_dir, f'{d}_seed_sensitivity.png'), dpi=150)
    plt.close(fig)

np.save(os.path.join(working_dir, 'summary_records.npy'), summary.reset_index().to_records(index=False), allow_pickle=True)
np.save(os.path.join(working_dir, 'cv_records.npy'), cv_df.to_records(index=False), allow_pickle=True)
np.save(os.path.join(working_dir, 'stats_records.npy'), stat_df.to_records(index=False), allow_pickle=True)

np.savez_compressed(
    os.path.join(working_dir, 'plot_data.npz'),
    results=results.to_records(index=False),
    cv=cv_df.to_records(index=False),
    stats=stat_df.to_records(index=False)
)

print('\nDone. Files saved to:', working_dir)