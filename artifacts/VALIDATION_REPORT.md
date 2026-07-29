# Validation report

Recorded output of [`scripts/validate_results.py`](../scripts/validate_results.py), run on
2026-07-28 against `workspace/run5/working/` and `workspace/run7/working/`.

Checks C1–C7 import only `numpy`, `pandas` and the standard library. They do not import LensKit or
any AutoRecLab code, and they reimplement 5-core filtering, so a defect in the generated pipeline
cannot be masked by the same defect in the checker. C8 is the one exception: it needs LensKit in
order to resolve the archived constructor, and is skipped when LensKit is not installed.

**Result: 13 of 13 checks passed, exit code 0.**

A machine-readable copy is written by `--json`:

```
python scripts/validate_results.py --json artifacts/validation_report.json
```

```
Independent validation of the AutoRecLab Team 5 results
expected completeness: 9 sub-experiments / 45 runs / 270 metric values

=== run5: workspace\run5\working\seed_sensitivity_results.csv ===
  [PASS] C1  45 algorithm-dataset-seed rows present  -- found 45
  [PASS] C2  every combination has exactly 5 distinct seeds  -- min=5 max=5 over 9 combinations
  [PASS] C3  no missing and no duplicated combinations  -- missing=0 unexpected=0 duplicated=0
  [PASS] C4  no NaN and no non-finite metric values  -- nan=0 inf=0
  [PASS] C5  all metrics at all cut-offs present, 270 values in [0,1]  -- columns=6 values=270 in_range=True
  [PASS] C6  recomputed means and CVs match Table 3 of the manuscript  -- all 9 sub-experiments match

=== run7: workspace\run7\working\seed_sensitivity_results.csv ===
  [PASS] C1  45 algorithm-dataset-seed rows present  -- found 45
  [PASS] C2  every combination has exactly 5 distinct seeds  -- min=5 max=5 over 9 combinations
  [PASS] C3  no missing and no duplicated combinations  -- missing=0 unexpected=0 duplicated=0
  [PASS] C4  no NaN and no non-finite metric values  -- nan=0 inf=0
  [PASS] C5  all metrics at all cut-offs present, 270 values in [0,1]  -- columns=6 values=270 in_range=True

=== 5-core interaction counts, recomputed from the raw data ===
  ml100k             recomputed  54,413 /    938 /  1,008   recorded  54,413 /    938 /  1,008
  amazon_videogames  recomputed 533,133 / 63,383 / 19,020   recorded 533,133 / 63,383 / 19,020
  lastfm             recomputed  52,551 /  1,090 /  3,646   recorded  52,551 /  1,090 /  3,646
  [PASS] C7  recomputed 5-core counts match run_manifest.json  -- all three datasets match

=== ItemKNN configuration, resolved against the installed LensKit ===
  run5: ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False)
  run7: ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False)
  as archived         : {'center': False, 'aggregate': 'weighted-average', 'use_ratings': True}
  feedback="implicit" : {'center': False, 'aggregate': 'sum', 'use_ratings': False}
  [PASS] C8  archived ItemKNN constructor is the reported misconfiguration  -- both nodes carry
             the identical constructor; it resolves to aggregate='weighted-average',
             use_ratings=True -- the explicit-feedback defaults, not the implicit ones

13 of 13 checks passed.
wrote artifacts\validation_report.json
```

**On the direction of C8.** The check passes when the misconfiguration *is* present, because that
is what the paper claims about the archived code. A failing C8 would mean the archived node is not
the one we describe. It resolves the constructor by instantiating `ItemItem` twice — once as
archived, once with `feedback='implicit'` — and reading back `center`, `aggregate` and
`use_ratings`, so the comparison comes from LensKit itself rather than from our reading of its
documentation.

## What this does and does not establish

**Establishes (evidence level 3 — independent logic).**
The expected completeness of 9 aggregated sub-experiments, 45 algorithm–dataset–seed runs and 270
individual metric values is fully met in both runs. Every value is present, unique and finite, and
all lie within the valid range [0, 1] for the metrics used. The means and coefficients of variation
printed in Table 3 of the manuscript follow from the archived per-seed file. The 5-core
preprocessing produces exactly the interaction, user and item counts the run recorded, for all
three datasets, when reimplemented from the raw files.

**Does not establish.**
That the numbers are *methodologically meaningful*. Completeness and internal consistency are
orthogonal to validity: the ItemKNN columns pass every check above while being invalid
measurements, because the algorithm was left in explicit-feedback mode
(see [RERUN_REPORT.md](RERUN_REPORT.md)). C6 confirms that Table 3 faithfully reports what the
node produced; it says nothing about whether what the node produced measures what it claims to.

**Note on C6 for Run 7.** The check is applied to Run 5 only, because Table 3 of the manuscript
reports Run 5. Run 7's aggregates were verified separately against
`workspace/run7/working/metric_summary_mean_std.csv` (ALS coefficients of variation 0.98%, 3.05%
and 4.78%, matching the "1.0–4.8%" stated in the paper).

**Note on the `.npy` files.** `*_predictions.npy` and `*_ground_truth.npy` in the working
directories do **not** contain predictions or ground truth. They hold counters only —
e.g. `{'seed': 1, 'algorithm': 'ALS', 'topn_users': 63383, 'num_recs': 633830}` and
`{'seed': 1, 'num_test': 129135}`. An independent recomputation of nDCG@10 or Precision@10 is
therefore not possible from the stored artefacts alone; it requires re-executing the pipeline, which
is what `scripts/rerun_selected_node.py` does.
