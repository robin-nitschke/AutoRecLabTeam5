# Re-execution report

Recorded output of [`scripts/rerun_selected_node.py`](../scripts/rerun_selected_node.py), run on
2026-07-28. Raw per-cell results: [`scripts/rerun_results.csv`](../scripts/rerun_results.csv).

The script re-executes the pipeline of the evaluated Run-5 node
([`selected_node_code.py`](run5-selected-node/selected_node_code.py),
tree-search iteration 9, reviewer score 72/100) in a separate process, and does four things with it:

1. compares the new metrics against the reported per-seed file (**evidence level 2**);
2. recomputes the metrics with an independent implementation — a plain per-user loop rather than
   the node's vectorised code (**level 3**);
3. checks the split for train/test leakage and the candidate sets for seen items;
4. repeats every ItemKNN cell under `feedback='implicit'`, the configuration LensKit 0.14.4
   documents for implicit data (**counterfactual**).

Coverage: MovieLens 100K and Last.FM at all five seeds, Amazon Video Games at seed 1 only —
its 63,383 evaluation users × ~19,000 candidates each make a full grid expensive, and the node's
candidate function is a lambda, which forces single-process execution. 44 cells in total.

---

## 0. Is the re-executed pipeline really the node's?

The script does not import the node file; it carries the node's pipeline functions as a
transcription, so that the loop around them can be driven per dataset and per seed and can write
one CSV row per cell. That transcription has to be faithful, or nothing below means anything.

We verified it by comparing abstract syntax trees, which ignores formatting, comments and naming
of the enclosing scope. Every pipeline function used in the re-run is **AST-identical** to its
counterpart in `run5-selected-node/selected_node_code.py`:

| Function in the script | Function in the node | Result |
|---|---|---|
| `k_core_filter` | `k_core_filter` | identical |
| `load_ml100k`, `load_amazon`, `load_lastfm` | same names | identical |
| `user_holdout_split` | `user_holdout_split` | identical |
| `build_candidate_map` | `build_candidate_map` | identical |
| `ensure_rank` | `ensure_rank` | identical |
| `node_metrics` | `precision_ndcg_at_k` | identical (renamed only) |

The algorithm constructors in the script's `ALGOS` registry are the ones the node's
`make_algorithms` returns:

```
ALS     : ImplicitMF(features=50, iterations=15, weight=40, method='cg', use_ratings=False)
ItemKNN : ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False)
Pop     : Popular()
```

Two node functions are deliberately **not** transcribed: `paired_stats` and
`plot_metric_variation`. They produce the statistics table and the figures, not the metric values,
and the re-run does not use them.

Reproduce the comparison with:

```python
import ast
def funcs(p):
    t = ast.parse(open(p, encoding='utf-8').read().replace('\r\n', '\n'))
    return {n.name: ast.dump(ast.parse(ast.unparse(n)))
            for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
a = funcs('scripts/rerun_selected_node.py')
b = funcs('artifacts/run5-selected-node/selected_node_code.py')
print({k: a[k] == b[k] for k in sorted(set(a) & set(b))})
```

The independent evidence points the same way: Most-Popular returns bit-exactly in all 11 cells and
ItemKNN in 10 of 11 (section 1), which a divergent transcription would not produce.

**Wording that follows from this.** The evidence level 2 claim in the paper is stated as a
re-execution of the node's pipeline, transcribed function-for-function from the node code and
verified identical — not as an execution of the node file itself.

---

## 1. Level 2 — does the reported number come back?

| Algorithm | Cells | Exact | max abs Δ nDCG@10 |
|---|---|---|---|
| Most-Popular | 11 | **11/11** | **0.0** |
| ItemKNN | 11 | **10/11** | 2.6 × 10⁻⁴ (Amazon only) |
| ALS | 11 | 0/11 | 8.9 × 10⁻³ |

**Most-Popular reproduces bit-exactly everywhere, and ItemKNN everywhere except Amazon.** Every
other reported value comes back to the last digit, which confirms that the archived node code is
the code that produced the reported numbers.

**The one ItemKNN exception is itself diagnostic.** On Amazon Video Games we obtain nDCG@10 =
0.001411 against a reported 0.001666. The reason is the misconfiguration described in section 4:
with every candidate scored identically at 1.0, the top-10 list is decided purely by the order in
which candidates are enumerated. `build_candidate_map` builds that order from a Python set
difference. MovieLens 100K and Last.FM identify items by **integers**, which hash to themselves, so
set iteration order is stable across processes and the result reproduces exactly. Amazon identifies
items by **string ASINs**, whose set order depends on the per-process string hash seed — so the
metric is not reproducible across processes at all. Most-Popular is unaffected because its scores
are genuinely distinct (17 distinct values on Amazon).

**ALS does not reproduce**, and the reason is a defect in the generated code rather than in our
re-run. `ImplicitMF.__init__` takes an `rng_spec` argument that defaults to `None`, and the model
initialises its factor matrices with `self.rng.standard_normal(...)`. The generated code passes no
`rng_spec`, so ALS is seeded only through the data split, not through the model.

| Dataset | Seed | Reported | Re-run | Rel. deviation |
|---|---|---|---|---|
| ml100k | 1 | 0.207318 | 0.210608 | 1.59% |
| ml100k | 7 | 0.201334 | 0.199412 | 0.95% |
| ml100k | 21 | 0.203834 | 0.200110 | 1.83% |
| ml100k | 42 | 0.210524 | 0.214189 | 1.74% |
| ml100k | 84 | 0.215969 | 0.207197 | 4.06% |
| lastfm | 1 | 0.117679 | 0.126572 | 7.56% |
| lastfm | 7 | 0.121993 | 0.118447 | 2.91% |
| lastfm | 21 | 0.119151 | 0.126626 | 6.27% |
| lastfm | 42 | 0.122565 | 0.123036 | 0.38% |
| lastfm | 84 | 0.119897 | 0.116748 | 2.63% |
| amazon | 1 | 0.065863 | 0.064487 | 2.09% |

(ALS uses float item factors, so its scores are effectively never tied and the ordering issue above
does not apply to it; its deviation comes purely from the unseeded initialisation.)

Aggregates are far more stable than individual cells: the five-seed mean of ALS nDCG@10 differs by
0.72% on MovieLens 100K (0.2078 vs 0.2063) and 1.69% on Last.FM (0.1203 vs 0.1223). Table 3 of the
paper is therefore reproducible as a set of means; its individual per-seed values are not.

The prompt asked for reproducible execution with explicit random seeds, and this requirement was in
the Planner's requirement list. The generated code satisfies it for the split and not for the model.
Neither the reviewer nor our own earlier inspection caught this.

## 2. Level 3 — is the metric code correct?

We reimplemented Precision@k and nDCG@k as a straightforward per-user loop and compared against the
node's vectorised implementation on every cell:

| Metric | Max abs difference over 44 cells |
|---|---|
| Precision@1, @5, @10 | **0.0** |
| nDCG@1, @5 | **0.0** |
| nDCG@10 | 1.1 × 10⁻¹⁶ (floating-point noise) |

**The node's metric computation is correct.** Precision, DCG, the IDCG normalisation over
`min(|relevant|, k)`, and the aggregation across users all check out, on all 33 node cells, which
covers all nine algorithm–dataset combinations. Whatever is wrong with the ItemKNN numbers is not
wrong in the measurement.

Held against the **reported** per-seed file rather than against the re-run, our independent
implementation returns the published values exactly (to within 5 × 10⁻⁷ on all six metric columns)
for **5 of the 9 combinations**:

| Combination | Independent implementation matches the reported file |
|---|---|
| Most-Popular on ML-100K, Last.FM, Amazon | **yes**, all three |
| ItemKNN on ML-100K, Last.FM | **yes** |
| ItemKNN on Amazon | no — hash-order tie-breaking (section 1) |
| ALS on all three datasets | no — unseeded `ImplicitMF` (section 1) |

The four that differ are exactly the non-deterministic cells. Where the pipeline is deterministic,
an independently written metric implementation reproduces the published number.

## 3. Split integrity

| Check | Result |
|---|---|
| Train/test interaction pairs overlapping, per cell | **0** in all 44 cells |
| Candidate sets containing items seen in training (probe: first 200 evaluation users) | **0** in all 44 cells |

No leakage between training and test data, and the candidate generation correctly excludes items a
user has already interacted with.

## 4. The ItemKNN misconfiguration

### The degeneracy is total

| Dataset | Seed | Recommendations | Distinct scores | Score std |
|---|---|---|---|---|
| ml100k | 1, 7, 21, 42, 84 | 9,380 each | **1** | 0.0 |
| lastfm | 1, 7, 21, 42, 84 | 10,900 each | **1** | 0.0 |
| amazon | 1 | 633,830 | **1** | 0.0 |

Every recommendation in every ItemKNN cell receives the identical score of exactly 1.0. The
top-10 list is decided entirely by tie-breaking; the similarity model contributes nothing.

**Why.** `ItemItem`'s `feedback` argument defaults to `'explicit'`, which sets
`aggregate='weighted-average'` and `use_ratings=True`. The node overrides only `center`. The
training frame has no rating column, so `sparse_ratings` enters every interaction as unary 1.0, and
`_predict_weighted_average` computes Σ(1·sⱼ)/Σ|sⱼ| = 1.0 for every item with at least one rated
neighbour. Verified by instantiation:

```
node configuration     : center=False  aggregate=weighted-average  use_ratings=True
documented implicit    : center=False  aggregate=sum               use_ratings=False
```

The identity follows algebraically from unary input, and re-execution confirms it on all three datasets.

### What the correct configuration gives (mean nDCG@10 over five seeds)

| | MovieLens 100K | Amazon VG (seed 1) | Last.FM |
|---|---|---|---|
| ItemKNN, as evaluated | 0.0433 | 0.0014 | 0.0045 |
| ItemKNN, `feedback='implicit'` | **0.2521** | 0.0585 | 0.1125 |
| Factor | **5.8×** | **41.4×** | **24.8×** |
| Most-Popular (reference) | 0.1563 | 0.0183 | 0.0568 |
| ALS (reference) | 0.2063 | **0.0645** | **0.1223** |
| **Corrected ranking** | **ItemKNN > ALS > Pop** | **ALS > ItemKNN > Pop** | **ALS > ItemKNN > Pop** |

Precision@10 moves the same way: 5.0× on MovieLens 100K and 18.1× on Last.FM.

**The reported ordering ALS > Pop > ItemKNN holds on none of the three datasets** once ItemKNN is
configured correctly. It is an artefact of the misconfiguration.

### Independent corroboration from Run 4

Run 4's generated code tries `ItemItem(nnbrs=20, feedback='implicit')` first
([`workspace/run4/runfile.py:183`](../workspace/run4/runfile.py)) and therefore configured the
algorithm correctly. Its MovieLens 100K and Last.FM inputs are identical to Run 5's
(54,413/938/1,008 and 52,551/1,090/3,646 interactions/users/items).

| nDCG@10 | Run 4 (correct config) | Our corrected re-run | Run 5 (as reported) |
|---|---|---|---|
| MovieLens 100K | 0.2557 | 0.2521 | 0.0433 |
| Last.FM | 0.1164 | 0.1125 | 0.0045 |

Run 4 and our independent re-measurement agree to within 2–4%, from two entirely separate code
paths. What Table 2 of the earlier manuscript version recorded as "run-to-run variation" was the
difference between a correct and an incorrect configuration.

## 5. What this does not cover

- The corrected-ItemKNN comparison covers all three datasets, but Amazon Video Games at one seed
  only, so its factor of 41.4× rests on a single split.
- The level-2 re-execution covers all five seeds on MovieLens 100K and Last.FM, seed 1 on Amazon.
- Nothing here reaches evidence level 4: no result was reproduced with an independent recommender
  implementation or against an external reference.
- Run 7 was not re-executed. It repeats the AutoRecLab pipeline rather than Run 5's code, and its
  node carries the same ItemKNN defect by inspection of its constructor.
