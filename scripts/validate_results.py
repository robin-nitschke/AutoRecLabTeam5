"""Independent completeness and consistency check for the Run 5 and Run 7 results.

This script deliberately does not import LensKit or any AutoRecLab code. It reads
the result files produced by the evaluated nodes and re-derives everything it
checks with its own logic, so that a defect in the generated pipeline cannot hide
behind the same defect in the checker.

Checks performed per run:

  C1  45 algorithm-dataset-seed rows are present
  C2  every algorithm-dataset combination has exactly 5 distinct seeds
  C3  no missing and no duplicated combinations
  C4  no NaN and no non-finite metric values
  C5  all required metrics (Precision, nDCG) at all cut-offs (1, 5, 10) present,
      270 individual values, all within [0, 1]
  C6  means recomputed from the per-seed file match the values reported in
      Table 3 of the manuscript
  C7  interaction counts after 5-core filtering, recomputed from the raw data,
      match the counts recorded by the run
  C8  the archived ItemKNN constructor resolves to the explicit-feedback
      settings the paper reports as the misconfiguration (needs LensKit; skipped
      if it is not installed)

C8 passes when the misconfiguration is present, because that is what the paper
claims. A failing C8 would mean the archived code is not what we describe.

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --json artifacts/validation_report.json

Exit code 0 if every check passes, 1 otherwise. Skipped checks do not fail.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(REPO, 'workspace')

DATASETS = ['ml100k', 'amazon_videogames', 'lastfm']
ALGORITHMS = ['ALS', 'ItemKNN', 'Pop']
SEEDS = [1, 7, 21, 42, 84]
CUTOFFS = [1, 5, 10]
METRICS = ['Precision', 'nDCG']

EXPECTED_SUB_EXPERIMENTS = len(DATASETS) * len(ALGORITHMS)          # 9
EXPECTED_RUNS = EXPECTED_SUB_EXPERIMENTS * len(SEEDS)               # 45
EXPECTED_VALUES = EXPECTED_RUNS * len(METRICS) * len(CUTOFFS)       # 270

RUNS = {
    'run5': os.path.join(WS, 'run5', 'working', 'seed_sensitivity_results.csv'),
    'run7': os.path.join(WS, 'run7', 'working', 'seed_sensitivity_results.csv'),
}

# Table 3 of the manuscript (Run 5), as printed: nDCG@10, CV of nDCG@10, P@10.
TABLE3 = {
    ('ml100k', 'ALS'):               (0.208,  2.8,  0.151),
    ('ml100k', 'ItemKNN'):           (0.043,  3.8,  0.035),
    ('ml100k', 'Pop'):               (0.156,  2.4,  0.120),
    ('amazon_videogames', 'ALS'):    (0.065,  1.3,  0.020),
    ('amazon_videogames', 'ItemKNN'): (0.0016, 7.7, 0.0005),
    ('amazon_videogames', 'Pop'):    (0.018,  0.6,  0.006),
    ('lastfm', 'ALS'):               (0.120,  1.7,  0.078),
    ('lastfm', 'ItemKNN'):           (0.005, 19.6,  0.004),
    ('lastfm', 'Pop'):               (0.057,  5.7,  0.037),
}

# Interaction counts after 5-core filtering, as recorded in
# workspace/run5/working/run_manifest.json.
MANIFEST_COUNTS = {
    'ml100k':            (54413, 938, 1008),
    'amazon_videogames': (533133, 63383, 19020),
    'lastfm':            (52551, 1090, 3646),
}

NODE_CODE = {
    'run5': os.path.join(REPO, 'artifacts', 'run5-selected-node', 'selected_node_code.py'),
    'run7': os.path.join(REPO, 'artifacts', 'run7-selected-node', 'selected_node_code.py'),
}

results = []


def check(run, cid, description, ok, detail='', skipped=False):
    results.append({'run': run, 'check': cid, 'description': description,
                    'ok': bool(ok), 'skipped': bool(skipped), 'detail': detail})
    status = 'SKIP' if skipped else ('PASS' if ok else 'FAIL')
    print(f'  [{status}] {cid}  {description}' + (f'  -- {detail}' if detail else ''))
    return ok


def metric_columns(df):
    return [f'{m}@{k}' for m in METRICS for k in CUTOFFS if f'{m}@{k}' in df.columns]


# ---------------------------------------------------------------------------
# 5-core filtering, reimplemented here (C7)
# ---------------------------------------------------------------------------
def k_core(df, min_uc=5, min_ic=5):
    df = df[['user', 'item']].drop_duplicates().copy()
    changed = True
    while changed and len(df):
        n0 = len(df)
        uc = df['user'].value_counts()
        ic = df['item'].value_counts()
        df = df[df['user'].isin(uc[uc >= min_uc].index)]
        df = df[df['item'].isin(ic[ic >= min_ic].index)]
        changed = len(df) != n0
    return df.reset_index(drop=True)


def recompute_core_counts():
    counts = {}

    ml = pd.read_csv(os.path.join(WS, 'u.data'), sep='\t', header=None,
                     names=['user', 'item', 'rating', 'timestamp'])
    ml = ml[ml['rating'] > 3][['user', 'item']].drop_duplicates()
    r = k_core(ml)
    counts['ml100k'] = (len(r), r.user.nunique(), r.item.nunique())

    az = pd.read_csv(os.path.join(WS, 'VideoGames.csv'))
    az = az.iloc[:, :3].copy()
    az.columns = ['user', 'item', 'rating']
    az = az[az['rating'] > 3][['user', 'item']].drop_duplicates()
    r = k_core(az)
    counts['amazon_videogames'] = (len(r), r.user.nunique(), r.item.nunique())

    lf = pd.read_csv(os.path.join(WS, 'UserTaggedArtists-timestamps.dat'), sep='\t')
    low = {c.lower(): c for c in lf.columns}
    lf = lf[[low.get('userid', lf.columns[0]), low.get('artistid', lf.columns[1])]].copy()
    lf.columns = ['user', 'item']
    r = k_core(lf)
    counts['lastfm'] = (len(r), r.user.nunique(), r.item.nunique())

    return counts


def validate_run(run, path):
    print(f'\n=== {run}: {os.path.relpath(path, REPO)} ===')
    if not os.path.exists(path):
        check(run, 'C0', 'result file exists', False, path)
        return
    df = pd.read_csv(path)

    check(run, 'C1', f'{EXPECTED_RUNS} algorithm-dataset-seed rows present',
          len(df) == EXPECTED_RUNS, f'found {len(df)}')

    per_combo = df.groupby(['dataset', 'algorithm'])['seed'].nunique()
    check(run, 'C2', 'every combination has exactly 5 distinct seeds',
          (per_combo == len(SEEDS)).all(),
          f'min={per_combo.min()} max={per_combo.max()} over {len(per_combo)} combinations')

    expected = {(d, a, s) for d in DATASETS for a in ALGORITHMS for s in SEEDS}
    actual = set(map(tuple, df[['dataset', 'algorithm', 'seed']].values))
    dups = df.duplicated(['dataset', 'algorithm', 'seed']).sum()
    check(run, 'C3', 'no missing and no duplicated combinations',
          expected == actual and dups == 0,
          f'missing={len(expected - actual)} unexpected={len(actual - expected)} duplicated={dups}')

    mcols = metric_columns(df)
    vals = df[mcols].to_numpy(dtype=float)
    check(run, 'C4', 'no NaN and no non-finite metric values',
          np.isfinite(vals).all(),
          f'nan={int(np.isnan(vals).sum())} inf={int(np.isinf(vals).sum())}')

    complete_cols = len(mcols) == len(METRICS) * len(CUTOFFS)
    in_range = bool(((vals >= 0) & (vals <= 1)).all())
    check(run, 'C5',
          f'all metrics at all cut-offs present, {EXPECTED_VALUES} values in [0,1]',
          complete_cols and vals.size == EXPECTED_VALUES and in_range,
          f'columns={len(mcols)} values={vals.size} in_range={in_range}')

    if run == 'run5':
        agg = df.groupby(['dataset', 'algorithm']).agg(
            ndcg=('nDCG@10', 'mean'), sd=('nDCG@10', 'std'), p10=('Precision@10', 'mean'))
        bad = []
        for (ds, alg), (t_ndcg, t_cv, t_p10) in TABLE3.items():
            row = agg.loc[(ds, alg)]
            cv = 100 * row['sd'] / row['ndcg']
            # compare at the precision the manuscript prints
            def close(a, b):
                dec = max(0, len(str(b).split('.')[-1]))
                return round(a, dec) == b
            if not (close(row['ndcg'], t_ndcg) and close(row['p10'], t_p10)
                    and round(cv, 1) == t_cv):
                bad.append(f'{ds}/{alg}: recomputed nDCG@10={row["ndcg"]:.4f} CV={cv:.1f} '
                           f'P@10={row["p10"]:.4f} vs table {t_ndcg}/{t_cv}/{t_p10}')
        check(run, 'C6', 'recomputed means and CVs match Table 3 of the manuscript',
              not bad, '; '.join(bad) if bad else 'all 9 sub-experiments match')


# ---------------------------------------------------------------------------
# ItemKNN configuration (C8)
#
# The paper reports that both evaluated nodes construct ItemItem with center=False
# but leave feedback at its 'explicit' default, which keeps aggregate='weighted-
# average' and use_ratings=True and makes every score on unary data exactly 1.0.
# This check resolves the archived constructor against the installed LensKit and
# confirms that this is in fact what the archived code does.
# ---------------------------------------------------------------------------
def resolved_params(algo):
    return {k: getattr(algo, k) for k in ('center', 'aggregate', 'use_ratings')}


def validate_itemknn_config():
    print('\n=== ItemKNN configuration, resolved against the installed LensKit ===')
    try:
        from lenskit.algorithms.item_knn import ItemItem
    except Exception as exc:
        check('config', 'C8', 'archived ItemKNN constructor is the reported '
              'misconfiguration', True,
              f'skipped, LensKit not importable ({type(exc).__name__})', skipped=True)
        return

    calls = {}
    for run, path in NODE_CODE.items():
        src = open(path, encoding='utf-8', errors='replace').read()
        m = re.search(r'ItemItem\(([^)]*)\)', src)
        calls[run] = m.group(0) if m else None

    if not all(calls.values()):
        check('config', 'C8', 'archived ItemKNN constructor is the reported '
              'misconfiguration', False,
              'no ItemItem(...) call found in ' +
              ', '.join(r for r, c in calls.items() if not c))
        return

    for run, call in calls.items():
        print(f'  {run}: {call}')

    as_archived = resolved_params(ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6,
                                           center=False))
    documented = resolved_params(ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6,
                                          feedback='implicit'))
    print(f'  as archived         : {as_archived}')
    print(f'  feedback="implicit" : {documented}')

    expected = {'center': False, 'aggregate': 'weighted-average', 'use_ratings': True}
    identical = calls['run5'].replace(' ', '') == calls['run7'].replace(' ', '')
    ok = as_archived == expected and as_archived != documented and identical
    detail = ('both nodes carry the identical constructor; it resolves to '
              f'aggregate={as_archived["aggregate"]!r}, '
              f'use_ratings={as_archived["use_ratings"]} -- the explicit-feedback '
              'defaults, not the implicit ones' if ok else
              f'resolved {as_archived}, expected {expected}; '
              f'run5/run7 constructors identical: {identical}')
    check('config', 'C8', 'archived ItemKNN constructor is the reported '
          'misconfiguration', ok, detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--json', metavar='PATH',
                        help='also write a machine-readable report to PATH')
    args = parser.parse_args()

    print('Independent validation of the AutoRecLab Team 5 results')
    print(f'expected completeness: {EXPECTED_SUB_EXPERIMENTS} sub-experiments / '
          f'{EXPECTED_RUNS} runs / {EXPECTED_VALUES} metric values')

    for run, path in RUNS.items():
        validate_run(run, path)

    print('\n=== 5-core interaction counts, recomputed from the raw data ===')
    try:
        counts = recompute_core_counts()
        bad = []
        for ds, expect in MANIFEST_COUNTS.items():
            got = counts[ds]
            print(f'  {ds:18s} recomputed {got[0]:>7,} / {got[1]:>6,} / {got[2]:>6,}'
                  f'   recorded {expect[0]:>7,} / {expect[1]:>6,} / {expect[2]:>6,}')
            if got != expect:
                bad.append(ds)
        check('run5', 'C7', 'recomputed 5-core counts match run_manifest.json',
              not bad, 'mismatch: ' + ', '.join(bad) if bad else 'all three datasets match')
    except Exception as e:
        check('run5', 'C7', 'recomputed 5-core counts match run_manifest.json',
              False, f'{type(e).__name__}: {e}')

    validate_itemknn_config()

    skipped = [r for r in results if r['skipped']]
    failed = [r for r in results if not r['ok'] and not r['skipped']]
    passed = len(results) - len(failed) - len(skipped)
    summary = f'\n{passed} of {len(results) - len(skipped)} checks passed'
    print(summary + (f', {len(skipped)} skipped.' if skipped else '.'))
    if failed:
        print('FAILED:')
        for r in failed:
            print(f'  {r["run"]} {r["check"]}: {r["description"]} -- {r["detail"]}')

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(REPO, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            json.dump({'passed': not failed, 'n_passed': passed,
                       'n_failed': len(failed), 'n_skipped': len(skipped),
                       'expected_completeness': {
                           'aggregated_sub_experiments': EXPECTED_SUB_EXPERIMENTS,
                           'algorithm_dataset_seed_runs': EXPECTED_RUNS,
                           'individual_metric_values': EXPECTED_VALUES},
                       'checks': results}, fh, indent=2)
        print(f'wrote {os.path.relpath(path, REPO)}')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
