import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Determinism controls set before heavy numeric imports where feasible.
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from lenskit import batch
from lenskit.algorithms.als import ImplicitMF

np.random.seed(0)

experiment_data = {
    'ml100k': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'timestamps': []
    }
}

cols = ['user', 'item', 'rating', 'timestamp']
path = os.path.join(os.getcwd(), 'u.data')
ratings = pd.read_csv(path, sep='\t', names=cols)

imp = ratings.loc[ratings['rating'] > 3, ['user', 'item']].drop_duplicates().copy()
while True:
    ucnt = imp.groupby('user').size()
    icnt = imp.groupby('item').size()
    good_users = set(ucnt[ucnt >= 5].index)
    good_items = set(icnt[icnt >= 5].index)
    new_imp = imp[imp['user'].isin(good_users) & imp['item'].isin(good_items)].copy()
    if len(new_imp) == len(imp):
        break
    imp = new_imp
all_items = set(imp['item'].unique().tolist())

def user_holdout_split(data, seed, test_frac=0.2):
    rng = np.random.RandomState(seed)
    trains, tests = [], []
    for u, udf in data.groupby('user', sort=False):
        items = udf['item'].to_numpy().copy()
        rng.shuffle(items)
        n_test = max(1, int(np.ceil(len(items) * test_frac)))
        n_test = min(n_test, len(items) - 1)
        test_items = items[:n_test]
        train_items = items[n_test:]
        trains.append(pd.DataFrame({'user': u, 'item': train_items, 'rating': 1.0}))
        tests.append(pd.DataFrame({'user': u, 'item': test_items, 'rating': 1.0}))
    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)

def dcg_at_k(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum())

def evaluate_topk(recs, test_df, users, k=10):
    truth = test_df.groupby('user')['item'].apply(set).to_dict()
    rec_groups = {u: g.sort_values('rank')['item'].tolist()[:k] for u, g in recs.groupby('user')} if len(recs) else {}
    eval_users = np.array(sorted(set(users).intersection(rec_groups.keys())))
    ndcgs, precs = [], []
    for u in eval_users:
        gt = truth.get(u, set())
        items = rec_groups.get(u, [])
        hits = [1.0 if i in gt else 0.0 for i in items]
        precs.append(sum(hits) / k)
        idcg = dcg_at_k(np.ones(min(len(gt), k)), k)
        ndcgs.append(dcg_at_k(hits, k) / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0, float(np.mean(precs)) if precs else 0.0, truth, eval_users

seeds = [1, 2, 3]
rows = []

for seed in seeds:
    train, test = user_holdout_split(imp, seed=seed, test_frac=0.2)
    requested_users = np.sort(test['user'].unique())
    train_items = train.groupby('user')['item'].apply(set).to_dict()

    algo = ImplicitMF()
    algo.fit(train)

    def candidate_selector(user, item_set=all_items, seen=train_items):
        return list(item_set - seen.get(user, set()))

    recs = batch.recommend(algo, requested_users, 10, candidates=candidate_selector)
    if len(recs):
        recs = recs.sort_values(['user', 'rank']).reset_index(drop=True)

    val_loss = float('nan')
    ndcg10, p10, truth, eval_users = evaluate_topk(recs, test, requested_users, k=10)
    print(f'Seed {seed}: validation_loss = {val_loss:.4f}')
    print(f'Seed {seed}: nDCG@10 = {ndcg10:.4f}, Precision@10 = {p10:.4f}, users = {len(eval_users)}')

    rows.append({'seed': seed, 'nDCG@10': ndcg10, 'Precision@10': p10})
    experiment_data['ml100k']['metrics']['val'].append({'seed': seed, 'nDCG@10': ndcg10, 'Precision@10': p10, 'users': int(len(eval_users))})
    experiment_data['ml100k']['losses']['val'].append({'seed': seed, 'validation_loss': val_loss})
    experiment_data['ml100k']['predictions'].append(recs[['user', 'item', 'rank', 'score']].to_dict('records') if len(recs) else [])
    experiment_data['ml100k']['ground_truth'].append({int(u): sorted(list(items)) for u, items in truth.items()})
    experiment_data['ml100k']['timestamps'].append({'seed': seed})

results = pd.DataFrame(rows)
summary = pd.DataFrame([
    {'seed': 'mean', 'nDCG@10': results['nDCG@10'].mean(), 'Precision@10': results['Precision@10'].mean()},
    {'seed': 'std', 'nDCG@10': results['nDCG@10'].std(ddof=1), 'Precision@10': results['Precision@10'].std(ddof=1)}
])
out = pd.concat([results, summary], ignore_index=True)
out_path = os.path.join(working_dir, 'ml100k_implicitmf_results.csv')
out.to_csv(out_path, index=False)

print('\nAggregate results:')
print(summary.to_string(index=False))
print(f'\nSaved CSV to: {out_path}')

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
np.save(os.path.join(working_dir, 'ml100k_metrics_per_seed.npy'), results.to_dict(orient='records'))
np.save(os.path.join(working_dir, 'ml100k_summary.npy'), summary.to_dict(orient='records'))
