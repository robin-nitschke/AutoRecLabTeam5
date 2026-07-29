# Run 5 — selected node

This directory archives the AutoRecLab node whose output produced the Run 5 numbers reported in
the paper, together with everything needed to trace those numbers back to it.

All files are **copies**; the originals remain at their paths in `out/run5/` and `workspace/run5/`.

## Which node this is

| | |
|---|---|
| Run | 5, executed 2026-05-08 |
| Tree-search iteration | **9 of 10** |
| Node ID | **not available** — see below |
| Reviewer score | **72 / 100**, `is_satisfactory=False` |
| Why this node | It is the only node in Run 5 whose execution succeeded. The three draft nodes and iterations 1–8 and 10 were all flagged buggy, so AutoRecLab fell back to "Found no satisfactory node; Using best node instead" (`run_stdout.log:609`). |

**No node ID exists for Run 5.** AutoRecLab writes its tree-search state to the fixed path
`./out/save.pkl` ([treesearch/search.py:126-129](../../treesearch/search.py#L126-L129)), and that
file was overwritten by Run 7 on 2026-07-16. The node is therefore identified by its tree-search
iteration, its reviewer score, its position in the log, and the SHA-256 of its code. (Run 7, whose
`save.pkl` survives, does have a node ID — see `../run7-selected-node/`.)

### Proof that `selected_node_code.py` is the code that produced the numbers

1. **Hash.** The `code` field of the `return_plan_and_code` tool call embedded at
   `run_stdout.log:556` has SHA-256 `6bfc3cb04775f1bbb09a99251fdb30853f79228906aed15455a19d201f71e32c`,
   identical to `selected_node_code.py` (CRLF normalised, whitespace stripped; both 11,867 characters).
2. **Output filenames.** This node writes `seed_sensitivity_results.csv`,
   `seed_sensitivity_results_long.csv`, `seed_sensitivity_summary.csv`, `seed_effect_summary.csv`,
   `paired_stats.csv` and `*_seed_variation.png` — all present in `workspace/run5/working/`.

> **Do not cite `workspace/run5/runfile.py` as the source of these numbers.** That file is a
> different, later node (iteration 10, buggy). It constructs `ImplicitMF()` and `ItemItem()` with
> plain defaults and writes `seed_sensitivity_results_wide.csv`,
> `seed_sensitivity_results_long_tidy.csv` and `paired_stats_descriptive.csv` — none of which exist
> in `workspace/run5/working/`. It also never imports `matplotlib`, so it cannot have produced the
> plots that are there.

## Methodological status of this node

| Algorithm | Constructor in the node | Status |
|---|---|---|
| ALS | `ImplicitMF(features=50, iterations=15, weight=40, method='cg', use_ratings=False)` | valid — configured for implicit feedback |
| Pop | `Popular()` | valid — no feedback-mode configuration needed |
| ItemKNN | `ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False)` | **invalid** |

**The ItemKNN configuration is wrong.** LensKit 0.14.4's `ItemItem` defaults to
`feedback='explicit'`, which sets `aggregate='weighted-average'` and `use_ratings=True`; the node
overrides only `center`. Because the training frame carries no rating column, every interaction
enters as unary 1.0, so the weighted average `Σ(1·sim)/Σ|sim|` evaluates to **exactly 1.0 for every
scorable item**. We confirmed this by re-execution: every ItemKNN cell produces exactly **one**
distinct recommendation score. The resulting ranking is pure tie-breaking and carries no similarity
signal. The correct configuration is `feedback='implicit'`.

The three ItemKNN sub-experiments are therefore **technically completed but scientifically
invalid**; the six ALS and Pop sub-experiments are valid. See
[../RERUN_REPORT.md](../RERUN_REPORT.md) for the quantified effect.

## Files

| File | What it is |
|---|---|
| `selected_node_code.py` | The evaluated node's Python code, verbatim. Copy of `workspace/run5/runfile_iter9_bestnode.py`. |
| `NODE.json` | Machine-readable metadata: iteration, score, hashes, algorithm configuration, environment. |
| `run_stdout.log` | Complete AutoRecLab console log of Run 5, including the embedded node code (line 556), the reviewer score (line 561) and the final summary. Copy of `out/run5/run_stdout.log`. |
| `prompt.txt` | The natural-language prompt given to AutoRecLab. Copy of `prompt_run5_paperreplikation.txt`. |
| `config.toml` | Tree-search and model configuration in force for this run. Copy of the repository-root `config.toml`. |
| `pyproject.toml` | Dependency pins in force for this run, including `lenskit==0.14.4`. Copy of the repository-root `pyproject.toml`. |
| `reviewer_score.txt` | **Reviewer score and full per-requirement feedback**, extracted from the `NodeScore` record at `run_stdout.log:561`. Requirements 13 and 21, which the paper discusses, are quoted from here. |
| `per_seed_results.csv` | **Per-seed result file** — 45 rows, one per dataset × algorithm × seed, with Precision and nDCG at cut-offs 1, 5 and 10. This is the file every number in Table 3 derives from. |
| `per_seed_results_long.csv` | The same 45 runs in long format, one row per metric value (270 rows). |
| `aggregated_results.csv` | **Aggregated result table** — mean, standard deviation and coefficient of variation per dataset × algorithm × metric. |
| `seed_effect_summary.csv` | Minimum, maximum, standard deviation and range across the five seeds per dataset × algorithm × metric. |
| `statistical_output.csv` | **Statistical output** — paired *t*-tests between algorithms on Precision@10 and nDCG@10, the source of Table 4. Note that these rest on five seed-matched pairs and are exploratory. |
| `run_manifest.json` | Seeds, cut-offs and post-5-core interaction/user/item counts per dataset, as recorded by the run. |

## Environment

AutoRecLab Alpha (initial release, `main` branch), Team 5 fork with patches P1–P5 applied on top of
upstream commit `34444d3cc77127eb73075a44980056c11b566c12` ("chore: bump version to 1.0.0",
2026-01-30). LensKit 0.14.4, Python 3.11, Windows 11, backend model `gpt-5.4`.

## Reproducing

`scripts/validate_results.py` re-derives the completeness and aggregate checks from
`per_seed_results.csv` without importing LensKit. `scripts/rerun_selected_node.py` re-executes this
node's pipeline and compares the new numbers against `per_seed_results.csv`. See
[../../REPRODUCING.md](../../REPRODUCING.md).
