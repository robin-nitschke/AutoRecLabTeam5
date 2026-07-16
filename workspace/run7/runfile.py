import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from lenskit import batch
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.item_knn import ItemItem

SEEDS = [1, 7, 21, 42, 84]
KS = [1, 5, 10]
MAX_K = max(KS)
DATASET_KEYS = ['ml100k', 'amazon_videogames', 'lastfm']

experiment_data = {
    k: {'metrics': {'train': [], 'val': []}, 'losses': {'train': [], 'val': []}, 'predictions': [], 'ground_truth': []}
    for k in DATASET_KEYS
}


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing required file: {path}')


def k_core_filter(df, user_col='user', item_col='item', min_k=5):
    df = df[[user_col, item_col]].dropna().drop_duplicates().copy()
    if df.empty:
        raise ValueError('Input interactions are empty before k-core filtering.')
    while True:
        uc = df[user_col].value_counts()
        ic = df[item_col].value_counts()
        keep_u = uc[uc >= min_k].index
        keep_i = ic[ic >= min_k].index
        new_df = df[df[user_col].isin(keep_u) & df[item_col].isin(keep_i)]
        if len(new_df) == len(df):
            break
        df = new_df
        if df.empty:
            raise ValueError('Dataset became empty during 5-core filtering.')
    return df.reset_index(drop=True)


def load_ml100k(path='u.data'):
    require_file(path)
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    if not {'user', 'item', 'rating'}.issubset(df.columns):
        raise ValueError('MovieLens schema mismatch.')
    df = df[df['rating'] > 3][['user', 'item']]
    return k_core_filter(df)


def load_amazon(path='VideoGames.csv'):
    require_file(path)
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    ucol = cols.get('user_id') or cols.get('userid') or cols.get('reviewerid')
    icol = cols.get('item_id') or cols.get('asin')
    rcol = cols.get('rating') or cols.get('overall')
    if not (ucol and icol and rcol):
        raise ValueError(f'Amazon schema mismatch. Found columns: {list(raw.columns)}')
    df = raw[[ucol, icol, rcol]].rename(columns={ucol: 'user', icol: 'item', rcol: 'rating'})
    df = df[df['rating'] > 3][['user', 'item']]
    return k_core_filter(df)


def load_lastfm(path='UserTaggedArtists-timestamps.dat'):
    require_file(path)
    df = pd.read_csv(path, sep='\t')
    cols = {c.lower(): c for c in df.columns}
    ucol = cols.get('userid')
    icol = cols.get('artistid')
    if not (ucol and icol):
        raise ValueError(f'LastFM schema mismatch. Found columns: {list(df.columns)}')
    out = df[[ucol, icol]].rename(columns={ucol: 'user', icol: 'item'})
    return k_core_filter(out)


def user_holdout_split(df, seed=42, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train_parts, test_parts, eval_users = [], [], []
    for user, udf in df.groupby('user', sort=False):
        n = len(udf)
        if n < 2:
            continue
        n_test = max(1, int(np.floor(n * test_frac)))
        n_test = min(n_test, n - 1)
        idx = np.arange(n)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True
        tr, te = udf.iloc[~mask], udf.iloc[mask]
        if len(tr) > 0 and len(te) > 0:
            train_parts.append(tr)
            test_parts.append(te)
            eval_users.append(user)
    if not train_parts or not test_parts:
        raise ValueError('No users support the holdout protocol after preprocessing.')
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test, len(eval_users)


def build_algorithms():
    return {
        'ALS': Recommender.adapt(ImplicitMF()),
        'ItemKNN': Recommender.adapt(ItemItem()),
        'Pop': Recommender.adapt(Popular())
    }


def _metrics_from_recs(rec_df, truth, ks):
    by_user = rec_df.groupby('user')['item'].apply(list).to_dict() if len(rec_df) else {}
    rows = []
    for k in ks:
        precs, ndcgs = [], []
        for u, rel in truth.items():
            ranked = by_user.get(u, [])[:k]
            hits = sum(1 for it in ranked if it in rel)
            precs.append(hits / k)
            dcg = sum((1.0 / np.log2(i + 2)) for i, it in enumerate(ranked) if it in rel)
            ideal = min(len(rel), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        rows.append((f'Precision@{k}', 'Precision', k, float(np.mean(precs)) if precs else np.nan))
        rows.append((f'nDCG@{k}', 'nDCG', k, float(np.mean(ndcgs)) if ndcgs else np.nan))
    return rows, by_user


def evaluate_algo(algo, train, test, ks):
    algo.fit(train)
    users = test['user'].drop_duplicates().tolist()
    try:
        recs = batch.recommend(algo, users, MAX_K, candidates=None)
    except TypeError:
        recs = batch.recommend(algo, users, MAX_K)
    train_items = train.groupby('user')['item'].apply(set).to_dict()
    truth = test.groupby('user')['item'].apply(set).to_dict()
    recs = recs[~recs.apply(lambda r: r['item'] in train_items.get(r['user'], set()), axis=1)].copy()
    recs['rank'] = recs.groupby('user').cumcount() + 1
    recs = recs[recs['rank'] <= MAX_K]
    metric_rows, by_user = _metrics_from_recs(recs[['user', 'item', 'rank']], truth, ks)
    return metric_rows, recs[['user', 'item', 'rank']], truth, by_user


def ci95(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


datasets = {}
for name, loader, path in [
    ('ml100k', load_ml100k, 'u.data'),
    ('amazon_videogames', load_amazon, 'VideoGames.csv'),
    ('lastfm', load_lastfm, 'UserTaggedArtists-timestamps.dat')
]:
    try:
        datasets[name] = loader(path)
        df = datasets[name]
        print(f'Loaded {name}: {len(df)} interactions, {df.user.nunique()} users, {df.item.nunique()} items')
    except Exception as e:
        print(f'Failed loading {name}: {e}')
        datasets[name] = pd.DataFrame(columns=['user', 'item'])

long_rows = []
run_rows = []
for dname, df in datasets.items():
    if df.empty:
        continue
    for seed in SEEDS:
        try:
            train, test, n_eval = user_holdout_split(df, seed=seed, test_frac=0.2)
        except Exception as e:
            print(f'Split failed for {dname} seed={seed}: {e}')
            continue
        for aname, algo in build_algorithms().items():
            timestamp = pd.Timestamp.now().isoformat()
            print(f'Epoch {seed}: validation_loss = {float("nan"):.4f}')
            try:
                metric_rows, recs, truth, by_user = evaluate_algo(algo, train, test, KS)
                metric_map = {}
                for metric_label, metric_name, k, score in metric_rows:
                    long_rows.append({
                        'dataset': dname, 'algorithm': aname, 'seed': seed,
                        'metric_name': metric_name, 'k': k, 'score': score,
                        'evaluation_user_count': n_eval, 'timestamp': timestamp
                    })
                    metric_map[metric_label] = score
                run_rows.append({'dataset': dname, 'algorithm': aname, 'seed': seed, 'evaluation_user_count': n_eval, 'timestamp': timestamp, **metric_map})
                experiment_data[dname]['metrics']['train'].append({'seed': seed, 'algorithm': aname, 'n_train': len(train), 'timestamp': timestamp})
                experiment_data[dname]['metrics']['val'].append({'seed': seed, 'algorithm': aname, 'metrics': metric_map, 'evaluation_user_count': n_eval, 'timestamp': timestamp})
                experiment_data[dname]['losses']['train'].append({'seed': seed, 'algorithm': aname, 'loss': np.nan, 'timestamp': timestamp})
                experiment_data[dname]['losses']['val'].append({'seed': seed, 'algorithm': aname, 'loss': np.nan, 'timestamp': timestamp})
                experiment_data[dname]['predictions'].append({'seed': seed, 'algorithm': aname, 'rows': recs.to_dict('records')})
                experiment_data[dname]['ground_truth'].append({'seed': seed, 'algorithm': aname, 'truth': {str(k): list(v) for k, v in truth.items()}})
                print(dname, aname, seed, {m: round(v, 4) for m, v in metric_map.items()})
            except Exception as e:
                print(f'Run failed for {dname} {aname} seed={seed}: {e}')
                for metric_name in ['Precision', 'nDCG']:
                    for k in KS:
                        long_rows.append({
                            'dataset': dname, 'algorithm': aname, 'seed': seed,
                            'metric_name': metric_name, 'k': k, 'score': np.nan,
                            'evaluation_user_count': n_eval, 'timestamp': timestamp
                        })

results_long = pd.DataFrame(long_rows)
results_wide = pd.DataFrame(run_rows)
results_long.to_csv(os.path.join(working_dir, 'seed_sensitivity_results_long.csv'), index=False)
results_wide.to_csv(os.path.join(working_dir, 'seed_sensitivity_results_wide.csv'), index=False)
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'results_long_records.npy'), results_long.to_records(index=False), allow_pickle=True)
np.save(os.path.join(working_dir, 'results_wide_records.npy'), results_wide.to_records(index=False), allow_pickle=True)

summary = (results_long.groupby(['dataset', 'algorithm', 'metric_name', 'k'])['score']
           .agg(['mean', 'std', 'min', 'max', 'count'])
           .reset_index())
summary['cv'] = summary['std'] / summary['mean'].replace(0, np.nan)
summary['ci95'] = results_long.groupby(['dataset', 'algorithm', 'metric_name', 'k'])['score'].apply(ci95).values
summary.to_csv(os.path.join(working_dir, 'metric_summary_long.csv'), index=False)
print('\nSeed sensitivity summary:')
print(summary.round(4))

focus = summary[(summary['k'] == 10) & (summary['metric_name'].isin(['nDCG', 'Precision']))].copy()
focus['sensitivity_rank'] = focus.groupby('metric_name')['std'].rank(ascending=False, method='min')
focus.to_csv(os.path.join(working_dir, 'seed_sensitivity_focus_k10.csv'), index=False)

analysis_lines = []
for metric_name in ['nDCG', 'Precision']:
    sub = focus[focus['metric_name'] == metric_name].sort_values(['std', 'cv'], ascending=False)
    if len(sub):
        top = sub.iloc[0]
        low = sub.iloc[-1]
        analysis_lines.append(
            f"Most seed-sensitive for {metric_name}@10: {top['dataset']} / {top['algorithm']} (std={top['std']:.4f}, cv={top['cv']:.4f}); least sensitive: {low['dataset']} / {low['algorithm']} (std={low['std']:.4f}, cv={low['cv']:.4f})."
        )

counts = results_long[['dataset', 'seed', 'evaluation_user_count']].drop_duplicates().sort_values(['dataset', 'seed'])
counts.to_csv(os.path.join(working_dir, 'evaluation_user_counts.csv'), index=False)
print('\nEvaluation user counts per dataset/seed:')
print(counts)

print('\nShort statistical analysis:')
for line in analysis_lines:
    print(line)
with open(os.path.join(working_dir, 'short_analysis.txt'), 'w') as f:
    f.write('\n'.join(analysis_lines + ['\nEvaluation user counts:', counts.to_string(index=False)]))

np.savez_compressed(
    os.path.join(working_dir, 'plot_data.npz'),
    results_long=results_long.to_records(index=False),
    results_wide=results_wide.to_records(index=False),
    summary=summary.to_records(index=False),
    eval_counts=counts.to_records(index=False)
)

print('\nDone. Files saved to:', working_dir)