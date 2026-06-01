import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# LensKit imports (0.14.4-safe)
# -----------------------------
import lenskit

ALS = None
ItemKNN = None
Pop = None

def _try_imports():
    global ALS, ItemKNN, Pop
    try:
        from lenskit.algorithms.als import BiasedMF as ALS_1
        ALS = ALS_1
    except Exception:
        try:
            from lenskit.algorithms.als import ImplicitMF as ALS_1
            ALS = ALS_1
        except Exception:
            ALS = None
    try:
        from lenskit.algorithms.item_knn import ItemKNN as ItemKNN_1
        ItemKNN = ItemKNN_1
    except Exception:
        try:
            from lenskit.algorithms.item_knn import ItemItem as ItemKNN_1
            ItemKNN = ItemKNN_1
        except Exception:
            ItemKNN = None
    try:
        from lenskit.algorithms.popular import Popular as Pop_1
        Pop = Pop_1
    except Exception:
        try:
            from lenskit.algorithms.basic import Popular as Pop_1
            Pop = Pop_1
        except Exception:
            Pop = None

_try_imports()

np.random.seed(42)

experiment_data = {
    'MovieLens100K': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'seeds': [], 'algorithms': []},
    'AmazonVideoGames': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'seeds': [], 'algorithms': []},
    'LastFM': {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': [], 'seeds': [], 'algorithms': []},
}

# -----------------------------
# Data loading / preprocessing
# -----------------------------
def five_core(df):
    changed = True
    while changed:
        changed = False
        uc = df['user'].value_counts()
        ic = df['item'].value_counts()
        keep = df['user'].isin(uc[uc >= 5].index) & df['item'].isin(ic[ic >= 5].index)
        if keep.sum() < len(df):
            df = df.loc[keep].copy()
            changed = True
    return df.reset_index(drop=True)


def read_movielens(path):
    df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
    df = df[df['rating'] > 3].copy()
    df['rating'] = 1.0
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    return df[['user', 'item', 'rating', 'timestamp']].reset_index(drop=True)


def read_amazon(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    user_col = cols.get('reviewerid', cols.get('user_id', list(df.columns)[0]))
    item_col = cols.get('asin', cols.get('item_id', list(df.columns)[1]))
    rating_col = cols.get('overall', cols.get('rating', None))
    ts_col = cols.get('unixreviewtime', cols.get('timestamp', None))
    if rating_col is not None:
        df = df[df[rating_col] > 3].copy()
    df['rating'] = 1.0
    df['timestamp'] = df[ts_col] if ts_col is not None else np.arange(len(df))
    df['user'] = df[user_col].astype(str)
    df['item'] = df[item_col].astype(str)
    return df[['user', 'item', 'rating', 'timestamp']].reset_index(drop=True)


def read_lastfm(path):
    df = pd.read_csv(path, sep='\t', header=None, engine='python')
    if df.shape[1] >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ['user', 'item', 'timestamp']
    else:
        raise ValueError('Unexpected LastFM format')
    df['rating'] = 1.0
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    return df[['user', 'item', 'rating', 'timestamp']].reset_index(drop=True)


def make_holdout(df, seed=0, test_frac=0.2):
    users = df['user'].unique()
    tr_users, te_users = train_test_split(users, test_size=test_frac, random_state=seed)
    train = df[df['user'].isin(tr_users)].copy()
    test = df[df['user'].isin(te_users)].copy()
    # Keep the test users represented in training by moving one interaction per test user.
    moved = []
    for u, grp in test.groupby('user', sort=False):
        moved.append(grp.iloc[[0]])
    if moved:
        moved_df = pd.concat(moved, ignore_index=True)
        train = pd.concat([train, moved_df], ignore_index=True)
        test = test.drop(moved_df.index, errors='ignore')
    return train.reset_index(drop=True), test.reset_index(drop=True)

# -----------------------------
# Evaluation
# -----------------------------
def ndcg_at_k(recs, gt, k):
    topk = recs[:k]
    hits = np.array([1.0 if i in gt else 0.0 for i in topk], dtype=float)
    if len(hits) == 0:
        return 0.0
    dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
    ideal = min(len(gt), k)
    if ideal == 0:
        return 0.0
    idcg = np.sum(1.0 / np.log2(np.arange(2, ideal + 2)))
    return float(dcg / idcg)


def precision_at_k(recs, gt, k):
    topk = recs[:k]
    if k == 0:
        return 0.0
    return float(np.mean([1.0 if i in gt else 0.0 for i in topk]) if topk else 0.0)


def eval_rows(rows, ks=(1, 5, 10)):
    out = {f'ndcg@{k}': [] for k in ks}
    out.update({f'precision@{k}': [] for k in ks})
    for _, gt, recs in rows:
        for k in ks:
            out[f'ndcg@{k}'].append(ndcg_at_k(recs, gt, k))
            out[f'precision@{k}'].append(precision_at_k(recs, gt, k))
    return {m: float(np.mean(v)) for m, v in out.items()}

# -----------------------------
# Model fitting / recommendation
# -----------------------------
def _as_rating_df(df):
    # Keep normalization explicit: implicit feedback becomes rating=1.0.
    out = df[['user', 'item', 'rating']].copy()
    out['user'] = out['user'].astype(str)
    out['item'] = out['item'].astype(str)
    out['rating'] = out['rating'].astype(float)
    return out


def train_lenskit_model(name, train_df):
    ratings = _as_rating_df(train_df)
    if name == 'ALS':
        if ALS is None:
            raise ImportError('ALS algorithm unavailable in installed LensKit')
        try:
            model = ALS(features=32, iterations=15, reg=0.1)
        except Exception:
            model = ALS()
    elif name == 'ItemKNN':
        if ItemKNN is None:
            raise ImportError('ItemKNN algorithm unavailable in installed LensKit')
        try:
            model = ItemKNN(nnbrs=40)
        except Exception:
            model = ItemKNN()
    else:
        if Pop is None:
            raise ImportError('Pop algorithm unavailable in installed LensKit')
        model = Pop()
    model.fit(ratings)
    return model


def predict_lenskit(model, train_df, test_df, kmax=10):
    train_items = train_df.groupby('user')['item'].apply(set).to_dict()
    users = test_df['user'].unique().tolist()
    all_items = train_df['item'].astype(str).unique().tolist()
    rows = []
    for u in users:
        seen = train_items.get(u, set())
        cand = [i for i in all_items if i not in seen]
        rec_items = cand[:kmax]
        if hasattr(model, 'recommend'):
            try:
                rec = model.recommend(u, n=kmax, candidates=cand)
                if isinstance(rec, pd.DataFrame) and 'item' in rec.columns:
                    rec_items = rec['item'].astype(str).tolist()
                else:
                    rec_items = [str(x) for x in list(rec)]
            except Exception:
                rec_items = cand[:kmax]
        gt = set(test_df.loc[test_df['user'] == u, 'item'].astype(str))
        rows.append((u, gt, rec_items[:kmax]))
    return rows


def fit_predict_pop(train_df, test_df, kmax=10):
    pop = train_df['item'].value_counts()
    ranks = pop.index.astype(str).tolist()
    train_items = train_df.groupby('user')['item'].apply(set).to_dict()
    rows = []
    for u, grp in test_df.groupby('user', sort=False):
        gt = set(grp['item'].astype(str))
        seen = train_items.get(u, set())
        recs = [i for i in ranks if i not in seen]
        rows.append((u, gt, recs[:kmax]))
    return rows

# -----------------------------
# Load datasets
# -----------------------------
datasets = {
    'MovieLens100K': read_movielens(os.path.join(os.getcwd(), 'u.data')),
    'AmazonVideoGames': read_amazon(os.path.join(os.getcwd(), 'VideoGames.csv')),
    'LastFM': read_lastfm(os.path.join(os.getcwd(), 'UserTaggedArtists-timestamps.dat')),
}

for ds_name, df in list(datasets.items()):
    df = five_core(df)
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    datasets[ds_name] = df
    print(f'{ds_name}: {len(df)} interactions, {df.user.nunique()} users, {df.item.nunique()} items')

# -----------------------------
# Experiment loop
# -----------------------------
seeds = [11, 22, 33, 44, 55]
algorithms = ['ALS', 'ItemKNN', 'Pop']
results = []

for ds_name, df in datasets.items():
    for seed in seeds:
        train_df, test_df = make_holdout(df, seed=seed, test_frac=0.2)
        for alg in algorithms:
            t0 = time.time()
            try:
                if alg == 'Pop':
                    rows = fit_predict_pop(train_df, test_df, kmax=10)
                else:
                    model = train_lenskit_model(alg, train_df)
                    rows = predict_lenskit(model, train_df, test_df, kmax=10)
                metrics = eval_rows(rows, ks=(1, 5, 10))
                metrics['fit_time'] = time.time() - t0
                metrics['seed'] = seed
                metrics['algorithm'] = alg
                metrics['dataset'] = ds_name
                results.append(metrics)
                experiment_data[ds_name]['metrics']['val'].append(metrics)
                experiment_data[ds_name]['seeds'].append(seed)
                experiment_data[ds_name]['algorithms'].append(alg)
                experiment_data[ds_name]['predictions'].append(np.array(rows, dtype=object))
                experiment_data[ds_name]['ground_truth'].append(np.array([r[1] for r in rows], dtype=object))
                print(f"{ds_name} | seed={seed} | {alg} | nDCG@10={metrics['ndcg@10']:.4f} P@10={metrics['precision@10']:.4f}")
            except Exception as e:
                print(f'{ds_name} | seed={seed} | {alg} failed: {e}')

res_df = pd.DataFrame(results)
res_path = os.path.join(working_dir, 'all_results.csv')
res_df.to_csv(res_path, index=False)

# -----------------------------
# Statistical summary
# -----------------------------
print('\nSummary (mean ± std over seeds):')
for (ds, alg), grp in res_df.groupby(['dataset', 'algorithm']):
    print(f'\n{ds} - {alg}')
    for m in ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']:
        mu = grp[m].mean()
        sd = grp[m].std(ddof=1)
        cv = sd / mu if mu != 0 else np.nan
        print(f'  {m}: {mu:.4f} ± {sd:.4f} (CV={cv:.3f})')

print('\nSeed sensitivity (lower std = more stable):')
for ds in res_df['dataset'].unique():
    sub = res_df[res_df['dataset'] == ds]
    for m in ['ndcg@10', 'precision@10']:
        st = sub.groupby('algorithm')[m].std(ddof=1).sort_values()
        print(f'{ds} {m}: ' + ', '.join([f'{k}={v:.4f}' for k, v in st.items()]))

# Simple paired test across algorithms per dataset for nDCG@10 and P@10
print('\nShort statistical analysis (paired t-tests on seed-wise results):')
for ds in res_df['dataset'].unique():
    sub = res_df[res_df['dataset'] == ds].sort_values('seed')
    for m in ['ndcg@10', 'precision@10']:
        pivot = sub.pivot_table(index='seed', columns='algorithm', values=m)
        if {'ALS', 'ItemKNN', 'Pop'}.issubset(pivot.columns) and len(pivot) >= 2:
            a, b, c = pivot['ALS'], pivot['ItemKNN'], pivot['Pop']
            print(f'\n{ds} {m}:')
            print(f"  ALS vs ItemKNN: t={stats.ttest_rel(a, b, nan_policy='omit').statistic:.3f}, p={stats.ttest_rel(a, b, nan_policy='omit').pvalue:.3g}")
            print(f"  ALS vs Pop:     t={stats.ttest_rel(a, c, nan_policy='omit').statistic:.3f}, p={stats.ttest_rel(a, c, nan_policy='omit').pvalue:.3g}")
            print(f"  ItemKNN vs Pop: t={stats.ttest_rel(b, c, nan_policy='omit').statistic:.3f}, p={stats.ttest_rel(b, c, nan_policy='omit').pvalue:.3g}")

# Save arrays and data
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'results_matrix.npy'), res_df[['dataset', 'algorithm', 'seed', 'ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10']].to_records(index=False), allow_pickle=True)
np.save(os.path.join(working_dir, 'results_df.npy'), res_df.to_records(index=False), allow_pickle=True)

# Plot: mean nDCG@10 by dataset/algorithm
plot_df = res_df.groupby(['dataset', 'algorithm'])['ndcg@10'].agg(['mean', 'std']).reset_index()
for ds in plot_df['dataset'].unique():
    fig, ax = plt.subplots(figsize=(7, 4))
    sdf = plot_df[plot_df['dataset'] == ds].set_index('algorithm').reindex(algorithms)
    ax.bar(np.arange(len(algorithms)), sdf['mean'].values, yerr=sdf['std'].values, capsize=4)
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_xticklabels(algorithms)
    ax.set_ylabel('nDCG@10')
    ax.set_title(f'{ds}: nDCG@10 across algorithms')
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, f'{ds}_ndcg10_by_algorithm.png'), dpi=150)
    plt.close(fig)

print('\nSaved results to:', res_path)
