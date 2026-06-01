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
    'amazon_vg': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
}

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]


def k_core_filter(df, user_col='user', item_col='item', min_uc=5, min_ic=5):
    df = df[[user_col, item_col]].drop_duplicates().copy()
    changed = True
    while changed:
        n0 = len(df)
        uc = df.groupby(user_col).size()
        keep_u = uc[uc >= min_uc].index
        df = df[df[user_col].isin(keep_u)]
        ic = df.groupby(item_col).size()
        keep_i = ic[ic >= min_ic].index
        df = df[df[item_col].isin(keep_i)]
        changed = len(df) != n0
    return df.reset_index(drop=True)


def load_ml100k(path='u.data'):
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] > 3][['user', 'item']].copy()
    return k_core_filter(df)


def load_amazon(path='VideoGames.csv'):
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        ucol = cols.get('user_id') or cols.get('userid') or cols.get('reviewerid') or list(df.columns)[0]
        icol = cols.get('item_id') or cols.get('asin') or cols.get('productid') or list(df.columns)[1]
        rcol = cols.get('rating') or cols.get('overall') or list(df.columns)[2]
        df = df.rename(columns={ucol: 'user', icol: 'item', rcol: 'rating'})
    except Exception:
        df = pd.read_csv(path, header=None, names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] > 3][['user', 'item']].copy()
    return k_core_filter(df)


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    df = pd.read_csv(path, sep='\t')
    cols = {c.lower(): c for c in df.columns}
    ucol = cols.get('userid', list(df.columns)[0])
    icol = cols.get('artistid', list(df.columns)[1])
    df = df.rename(columns={ucol: 'user', icol: 'item'})[['user', 'item']].copy()
    return k_core_filter(df)


def user_holdout(df, seed=42, test_ratio=0.2):
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for _, g in df.groupby('user'):
        g = g.sample(frac=1.0, random_state=seed)
        n_test = max(1, int(np.ceil(len(g) * test_ratio)))
        test_parts.append(g.iloc[:n_test])
        train_parts.append(g.iloc[n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    # ensure all train users/items remain valid
    test = test[test['item'].isin(train['item'].unique())]
    tu = test['user'].unique()
    train = train[train['user'].isin(tu)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def precision_at_k(rec_items, true_items, k):
    if k == 0:
        return 0.0
    rec_k = rec_items[:k]
    if len(rec_k) == 0:
        return 0.0
    hits = sum(1 for x in rec_k if x in true_items)
    return hits / k


def ndcg_at_k(rec_items, true_items, k):
    rec_k = rec_items[:k]
    dcg = 0.0
    for i, it in enumerate(rec_k):
        if it in true_items:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(true_items), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recommend_for_users(algo, train, test, n=10):
    all_items = pd.Index(train['item'].unique())
    truth = test.groupby('user')['item'].apply(set).to_dict()
    users = sorted(truth.keys())
    recs = []
    for u in users:
        seen = set(train.loc[train['user'] == u, 'item'])
        candidates = all_items[~all_items.isin(list(seen))]
        if len(candidates) == 0:
            continue
        try:
            ru = batch.recommend(algo, [u], n, candidates=candidates)
        except Exception:
            continue
        if ru is None or len(ru) == 0:
            continue
        ru = ru[['user', 'item', 'score']].copy()
        recs.append(ru)
    if len(recs) == 0:
        return pd.DataFrame(columns=['user', 'item', 'score']), truth
    return pd.concat(recs, ignore_index=True), truth


def eval_recs(recs, truth, ks=(1, 5, 10)):
    rows = []
    for u, gt in truth.items():
        ru = recs[recs['user'] == u].sort_values('score', ascending=False)
        items = ru['item'].tolist()
        row = {'user': u}
        for k in ks:
            row[f'precision@{k}'] = precision_at_k(items, gt, k)
            row[f'ndcg@{k}'] = ndcg_at_k(items, gt, k)
        rows.append(row)
    return pd.DataFrame(rows)


def build_algorithms():
    return {
        'Pop': Popular(),
        'ItemKNN': ItemItem(20, min_nbrs=1, center=False),
        'ALS': ImplicitMF(50, iterations=10, weight=40)
    }


def run_dataset(name, df):
    all_results = []
    for seed in SEEDS:
        train, test = user_holdout(df, seed=seed, test_ratio=0.2)
        algos = build_algorithms()
        for algo_name, algo in algos.items():
            algo.fit(train)
            recs, truth = recommend_for_users(algo, train, test, n=max(KS))
            metrics = eval_recs(recs, truth, KS)
            val_loss = 1.0 - metrics['ndcg@10'].mean() if len(metrics) else 1.0
            print(f'Epoch {seed}: validation_loss = {val_loss:.4f}')
            row = {'dataset': name, 'algorithm': algo_name, 'seed': seed, 'n_users_eval': len(metrics)}
            for k in KS:
                row[f'precision@{k}'] = metrics[f'precision@{k}'].mean() if len(metrics) else 0.0
                row[f'ndcg@{k}'] = metrics[f'ndcg@{k}'].mean() if len(metrics) else 0.0
            all_results.append(row)
            experiment_data[name]['metrics']['val'].append({'seed': seed, 'algorithm': algo_name, **{k: row[k] for k in row if '@' in k}})
            experiment_data[name]['losses']['val'].append({'seed': seed, 'algorithm': algo_name, 'validation_loss': float(val_loss)})
            experiment_data[name]['predictions'].append(recs[['user', 'item', 'score']].to_dict('records'))
            experiment_data[name]['ground_truth'].append({str(u): list(v) for u, v in truth.items()})
    return pd.DataFrame(all_results)


def summarize_and_plot(results, dataset_name):
    metric_cols = [f'precision@{k}' for k in KS] + [f'ndcg@{k}' for k in KS]
    summary = results.groupby(['dataset', 'algorithm'])[metric_cols].agg(['mean', 'std']).reset_index()
    print('\nSummary for', dataset_name)
    print(summary.to_string(index=False))

    stats_rows = []
    for algo in results['algorithm'].unique():
        sub = results[(results['dataset'] == dataset_name) & (results['algorithm'] == algo)]
        for metric in metric_cols:
            groups = [sub[sub['seed'] == s][metric].values for s in SEEDS if len(sub[sub['seed'] == s]) > 0]
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                try:
                    fval, pval = stats.f_oneway(*groups)
                except Exception:
                    fval, pval = np.nan, np.nan
            else:
                fval, pval = np.nan, np.nan
            stats_rows.append({'dataset': dataset_name, 'algorithm': algo, 'metric': metric, 'anova_F': fval, 'anova_p': pval})
    stats_df = pd.DataFrame(stats_rows)
    print('\nSeed sensitivity stats for', dataset_name)
    print(stats_df.to_string(index=False))

    plot_metrics = ['precision@10', 'ndcg@10']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, plot_metrics):
        grp = results[results['dataset'] == dataset_name].groupby('algorithm')[metric].agg(['mean', 'std']).reset_index()
        ax.bar(grp['algorithm'], grp['mean'], yerr=grp['std'], capsize=4)
        ax.set_title(f'{dataset_name} {metric}')
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(1e-6, (grp['mean'] + grp['std']).max() * 1.2))
    plt.tight_layout()
    fig_path = os.path.join(working_dir, f'{dataset_name}_metrics_bar.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    np.save(os.path.join(working_dir, f'{dataset_name}_summary.npy'), summary.to_dict())
    np.save(os.path.join(working_dir, f'{dataset_name}_anova.npy'), stats_df.to_dict())
    return summary, stats_df


ml = load_ml100k('u.data')
amz = load_amazon('VideoGames.csv')
lfm = load_lastfm('UserTaggedArtists-timestamps.dat')

datasets = {'ml100k': ml, 'amazon_vg': amz, 'lastfm': lfm}
all_results = []
for name, df in datasets.items():
    print(f'Running dataset={name}, interactions={len(df)}, users={df.user.nunique()}, items={df.item.nunique()}')
    res = run_dataset(name, df)
    all_results.append(res)

results_df = pd.concat(all_results, ignore_index=True)
results_path = os.path.join(working_dir, 'all_results.csv')
results_df.to_csv(results_path, index=False)
print('\nPer-run results')
print(results_df.to_string(index=False))

for dname in datasets:
    summarize_and_plot(results_df, dname)

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.savez_compressed(os.path.join(working_dir, 'results_arrays.npz'), results=results_df.to_records(index=False))
print(f'\nSaved outputs to: {working_dir}')