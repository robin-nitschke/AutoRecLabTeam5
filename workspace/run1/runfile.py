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

experiment_data = {
    'ml100k': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'amazon_vg': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
    'lastfm': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []},
}

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]
MAX_K = max(KS)


def kcore_filter(df, user_col='user', item_col='item', min_uc=5, min_ic=5):
    df = df[[user_col, item_col]].drop_duplicates().copy()
    changed = True
    while changed and len(df):
        before = len(df)
        ucnt = df.groupby(user_col)[item_col].transform('count')
        df = df.loc[ucnt >= min_uc]
        icnt = df.groupby(item_col)[user_col].transform('count')
        df = df.loc[icnt >= min_ic]
        changed = len(df) != before
    df = df.reset_index(drop=True)
    df.columns = ['user', 'item']
    return df


def load_ml100k(path='u.data'):
    df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
    df = df.loc[df['rating'] > 3, ['user', 'item']].copy()
    return kcore_filter(df)


def load_amazon(path='VideoGames.csv'):
    try:
        df = pd.read_csv(path)
        cols = {str(c).lower(): c for c in df.columns}
        user_col = cols.get('userid') or cols.get('user_id') or cols.get('reviewerid') or df.columns[0]
        item_col = cols.get('itemid') or cols.get('asin') or cols.get('item_id') or df.columns[1]
        rating_col = cols.get('rating') or cols.get('overall') or df.columns[2]
        df = df[[user_col, item_col, rating_col]].copy()
        df.columns = ['user', 'item', 'rating']
    except Exception:
        df = pd.read_csv(path, header=None, names=['user', 'item', 'rating'])
    df = df.loc[pd.to_numeric(df['rating'], errors='coerce') > 3, ['user', 'item']].copy()
    return kcore_filter(df)


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    df = pd.read_csv(path, sep='\t', engine='python')
    cols = {str(c).lower(): c for c in df.columns}
    user_col = cols.get('userid', df.columns[0])
    item_col = cols.get('artistid', df.columns[1])
    df = df[[user_col, item_col]].copy()
    df.columns = ['user', 'item']
    return kcore_filter(df)


def user_holdout_split(df, seed=42, test_ratio=0.2):
    rng = np.random.default_rng(seed)
    trains, tests = [], []
    for _, udf in df.groupby('user', sort=False):
        udf = udf.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(udf)
        if n < 2:
            trains.append(udf)
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)
        idx = rng.choice(np.arange(n), size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        tests.append(udf.loc[mask])
        trains.append(udf.loc[~mask])
    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(columns=df.columns)
    return train[['user', 'item']], test[['user', 'item']]


def build_algorithms():
    return {
        'Pop': Popular(),
        'ItemKNN': ItemItem(20, center=False),
        'ALS': ImplicitMF(50, iterations=10, weight=40),
    }


def get_candidates(train_df, all_items, user):
    seen = set(train_df.loc[train_df['user'] == user, 'item'])
    return [i for i in all_items if i not in seen]


def dcg_at_k(rels):
    rels = np.asarray(rels, dtype=float)
    if rels.size == 0:
        return 0.0
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum())


def evaluate_algo(algo, train, test, ks=(1, 5, 10)):
    train_fit = train.copy()
    train_fit['rating'] = 1.0
    algo.fit(train_fit)
    users = sorted(set(test['user']).intersection(set(train['user'])))
    all_items = list(pd.Index(train['item'].unique()))
    truth = test.groupby('user')['item'].apply(set).to_dict()
    rows, rec_store = [], []
    for u in users:
        gt = truth.get(u, set())
        if not gt:
            continue
        cands = get_candidates(train, all_items, u)
        if not cands:
            continue
        try:
            recs = batch.recommend(algo, [u], MAX_K, candidates={u: cands})
        except Exception:
            recs = pd.DataFrame(columns=['user', 'item', 'score', 'rank'])
        rec_items = []
        if recs is not None and len(recs):
            recs = recs.sort_values('rank')
            rec_items = recs['item'].tolist()
            rec_store.append(recs.assign(dataset_user=u))
        for k in ks:
            topk = rec_items[:k]
            hits = np.array([1.0 if i in gt else 0.0 for i in topk], dtype=float)
            prec = float(hits.mean()) if len(topk) else 0.0
            idcg = dcg_at_k(np.ones(min(len(gt), k), dtype=float))
            ndcg = dcg_at_k(hits) / idcg if idcg > 0 else 0.0
            rows.append({'user': u, 'k': k, 'precision': prec, 'ndcg': ndcg})
    metrics = pd.DataFrame(rows)
    preds = pd.concat(rec_store, ignore_index=True) if rec_store else pd.DataFrame(columns=['user', 'item', 'score', 'rank'])
    return metrics, preds


def summarize_seed_effect(df_metrics):
    out = []
    for (dataset, algo, metric, k), g in df_metrics.groupby(['dataset', 'algorithm', 'metric', 'k']):
        vals = g['value'].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = float(std / mean) if mean != 0 else np.nan
        rng = float(np.max(vals) - np.min(vals)) if len(vals) else np.nan
        try:
            t_stat, t_p = stats.ttest_1samp(vals, popmean=mean) if len(vals) > 1 else (np.nan, np.nan)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        out.append({
            'dataset': dataset, 'algorithm': algo, 'metric': metric, 'k': k,
            'mean': mean, 'std': std, 'cv': cv, 'min': float(np.min(vals)), 'max': float(np.max(vals)),
            'range': rng, 'n_seeds': int(len(vals)), 'ttest_stat': t_stat, 'ttest_p': t_p
        })
    return pd.DataFrame(out)


datasets = {
    'ml100k': load_ml100k('u.data'),
    'amazon_vg': load_amazon('VideoGames.csv'),
    'lastfm': load_lastfm('UserTaggedArtists-timestamps.dat'),
}

all_results = []

for dname, df in datasets.items():
    print(f'Loaded {dname}: {len(df):,} interactions, {df.user.nunique():,} users, {df.item.nunique():,} items')
    for seed in SEEDS:
        train, test = user_holdout_split(df, seed=seed, test_ratio=0.2)
        val_loss = float(len(test) / max(len(train) + len(test), 1))
        print(f'Epoch {seed}: validation_loss = {val_loss:.4f}')
        experiment_data[dname]['losses']['val'].append({'seed': seed, 'validation_loss': val_loss, 'timestamp': pd.Timestamp.utcnow().isoformat()})
        experiment_data[dname]['ground_truth'].append({
            'seed': seed,
            'n_train': int(len(train)),
            'n_test': int(len(test)),
            'users_train': int(train['user'].nunique()),
            'users_test': int(test['user'].nunique())
        })
        for aname, algo in build_algorithms().items():
            metrics_df, pred_df = evaluate_algo(algo, train, test, ks=KS)
            if metrics_df.empty:
                continue
            agg = metrics_df.groupby('k')[['precision', 'ndcg']].mean().reset_index()
            for _, r in agg.iterrows():
                for metric in ['precision', 'ndcg']:
                    rec = {
                        'dataset': dname,
                        'algorithm': aname,
                        'seed': seed,
                        'metric': metric,
                        'k': int(r['k']),
                        'value': float(r[metric])
                    }
                    all_results.append(rec)
                    experiment_data[dname]['metrics']['val'].append({**rec, 'timestamp': pd.Timestamp.utcnow().isoformat()})
            experiment_data[dname]['predictions'].append({
                'algorithm': aname,
                'seed': seed,
                'n_predictions': int(len(pred_df)),
                'timestamp': pd.Timestamp.utcnow().isoformat()
            })
            msg = ', '.join([f"P@{int(r.k)}={r.precision:.4f}, nDCG@{int(r.k)}={r.ndcg:.4f}" for _, r in agg.iterrows()])
            print(f'{dname} | seed={seed} | {aname} -> {msg}')

results_df = pd.DataFrame(all_results)
results_path = os.path.join(working_dir, 'all_metrics.csv')
results_df.to_csv(results_path, index=False)

summary_df = summarize_seed_effect(results_df)
summary_path = os.path.join(working_dir, 'seed_effect_summary.csv')
summary_df.to_csv(summary_path, index=False)

plotting_available = True
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plotting_available = False
    print(f'Plotting skipped: {e}')

if plotting_available and not results_df.empty:
    for dname in datasets.keys():
        ddf = results_df.loc[results_df['dataset'] == dname]
        for metric in ['precision', 'ndcg']:
            fig, axes = plt.subplots(1, len(KS), figsize=(14, 4), sharey=False)
            axes = np.atleast_1d(axes)
            for ax, k in zip(axes, KS):
                sdf = ddf.loc[(ddf['metric'] == metric) & (ddf['k'] == k)]
                for aname, g in sdf.groupby('algorithm'):
                    g = g.sort_values('seed')
                    ax.plot(g['seed'], g['value'], marker='o', label=aname)
                ax.set_title(f'{dname} {metric}@{k}')
                ax.set_xlabel('seed')
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
            if len(axes):
                axes[0].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f'{dname}_{metric}_seed_variation.png'), dpi=150)
            plt.close(fig)

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.save(os.path.join(working_dir, 'results_array.npy'), results_df.to_records(index=False))
np.save(os.path.join(working_dir, 'summary_array.npy'), summary_df.to_records(index=False))

print('\nMean ± std across seeds:')
if not summary_df.empty:
    for _, r in summary_df.sort_values(['dataset', 'algorithm', 'metric', 'k']).iterrows():
        print(f"{r['dataset']} | {r['algorithm']} | {r['metric']}@{int(r['k'])}: {r['mean']:.4f} ± {r['std']:.4f} (CV={r['cv']:.4f}, range={r['range']:.4f})")

print(f'\nSaved metrics to: {results_path}')
print(f'Saved seed-effect summary to: {summary_path}')