import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import random
import numpy as np
import pandas as pd
from lenskit.algorithms.als import ImplicitMF
from lenskit import batch

experiment_data = {
    'ml100k_implicitmf': {
        'metrics': {'train': [], 'val': []},
        'losses': {'train': [], 'val': []},
        'predictions': [],
        'ground_truth': [],
        'timestamps': []
    }
}

def load_data(path='u.data'):
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    return df.loc[df['rating'] > 3, ['user', 'item']].reset_index(drop=True)

def iterative_kcore(df, k=5):
    cur = df[['user', 'item']].copy()
    while True:
        uc = cur.groupby('user').size()
        ic = cur.groupby('item').size()
        nxt = cur[cur['user'].isin(uc[uc >= k].index) & cur['item'].isin(ic[ic >= k].index)].copy()
        if len(nxt) == len(cur):
            return nxt.reset_index(drop=True)
        cur = nxt

def make_holdout(df, seed, test_ratio=0.2):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    train_parts, test_parts, skipped = [], [], []
    for user, udf in df.groupby('user', sort=False):
        n = len(udf)
        n_test = int(np.floor(n * test_ratio))
        if n < 2 or n_test < 1 or n - n_test < 1:
            skipped.append(user)
            continue
        idx = np.arange(n)
        rng.shuffle(idx)
        test_parts.append(udf.iloc[idx[:n_test]][['user', 'item']])
        train_parts.append(udf.iloc[idx[n_test:]][['user', 'item']])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=['user', 'item'])
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=['user', 'item'])
    return train, test, skipped

def dcg_at_k(rels, k=10):
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum())

def evaluate(recs, test, users, k=10):
    truth = test.groupby('user')['item'].apply(set).to_dict()
    ranked = recs.sort_values(['user', 'rank']).groupby('user')['item'].apply(list).to_dict()
    ndcgs, precs, preds, gts = [], [], [], []
    for u in users:
        rec_items = ranked.get(u, [])[:k]
        gt = truth.get(u, set())
        hits = [1 if i in gt else 0 for i in rec_items]
        precs.append(float(np.mean(hits)) if hits else 0.0)
        ideal = [1] * min(len(gt), k)
        ndcgs.append(dcg_at_k(hits, k) / dcg_at_k(ideal, k) if ideal else 0.0)
        preds.append((u, rec_items))
        gts.append((u, sorted(gt)))
    return float(np.mean(ndcgs)), float(np.mean(precs)), preds, gts

def recommend_excluding_train(model, train, users, n=10):
    users = list(users)
    train_items = train.groupby('user')['item'].apply(set).to_dict()
    all_items = pd.Index(train['item'].unique())
    candidates = {u: list(all_items.difference(pd.Index(list(train_items.get(u, set()))))) for u in users}
    recs = batch.recommend(model, users, n, candidates=candidates)
    return recs.sort_values(['user', 'rank']).reset_index(drop=True)

df = iterative_kcore(load_data('u.data'), k=5)
seeds = [1, 2, 3]
rows = []

for seed in seeds:
    train, test, skipped = make_holdout(df, seed, test_ratio=0.2)
    eval_users = sorted(set(train['user']).intersection(test['user']))
    train = train[train['user'].isin(eval_users)].reset_index(drop=True)
    test = test[test['user'].isin(eval_users)].reset_index(drop=True)

    random.seed(seed)
    np.random.seed(seed)
    algo = ImplicitMF()
    algo.fit(train)

    recs = recommend_excluding_train(algo, train, eval_users, n=10)
    ndcg, prec, preds, gts = evaluate(recs, test, eval_users, k=10)

    rows.append({
        'seed': seed,
        'users_evaluated': len(eval_users),
        'ndcg@10': ndcg,
        'precision@10': prec,
        'ndcg@10_mean': np.nan,
        'ndcg@10_std': np.nan,
        'precision@10_mean': np.nan,
        'precision@10_std': np.nan
    })
    experiment_data['ml100k_implicitmf']['metrics']['val'].append({'seed': seed, 'ndcg@10': ndcg, 'precision@10': prec})
    experiment_data['ml100k_implicitmf']['losses']['val'].append({'seed': seed, 'validation_loss': np.nan})
    experiment_data['ml100k_implicitmf']['predictions'].append(preds)
    experiment_data['ml100k_implicitmf']['ground_truth'].append(gts)
    experiment_data['ml100k_implicitmf']['timestamps'].append({'seed': seed, 'skipped_users': len(skipped), 'evaluated_users': len(eval_users)})
    print(f'Epoch {seed}: validation_loss = {np.nan:.4f}')
    print(f'Seed {seed}: nDCG@10 = {ndcg:.6f}, Precision@10 = {prec:.6f}')

res = pd.DataFrame(rows)
ndcg_mean = res['ndcg@10'].mean()
ndcg_std = res['ndcg@10'].std(ddof=1)
prec_mean = res['precision@10'].mean()
prec_std = res['precision@10'].std(ddof=1)
agg = pd.DataFrame([{
    'seed': 'aggregate',
    'users_evaluated': res['users_evaluated'].sum(),
    'ndcg@10': np.nan,
    'precision@10': np.nan,
    'ndcg@10_mean': ndcg_mean,
    'ndcg@10_std': ndcg_std,
    'precision@10_mean': prec_mean,
    'precision@10_std': prec_std
}])
out = pd.concat([res, agg], ignore_index=True)
out_csv = os.path.join(working_dir, 'ml100k_lenskit_implicitmf_reproducibility.csv')
out.to_csv(out_csv, index=False)

print(f'Aggregated nDCG@10: mean = {ndcg_mean:.6f}, std = {ndcg_std:.6f}')
print(f'Aggregated Precision@10: mean = {prec_mean:.6f}, std = {prec_std:.6f}')

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data, allow_pickle=True)
np.save(os.path.join(working_dir, 'ml100k_metrics.npy'), res[['ndcg@10', 'precision@10']].to_numpy())
np.save(os.path.join(working_dir, 'ml100k_predictions.npy'), np.array(experiment_data['ml100k_implicitmf']['predictions'], dtype=object), allow_pickle=True)
np.save(os.path.join(working_dir, 'ml100k_ground_truth.npy'), np.array(experiment_data['ml100k_implicitmf']['ground_truth'], dtype=object), allow_pickle=True)
np.save(os.path.join(working_dir, 'ml100k_timestamps.npy'), np.array(experiment_data['ml100k_implicitmf']['timestamps'], dtype=object), allow_pickle=True)
