# Artefact manifest

This repository contains artefacts from seven AutoRecLab runs. Most of them document **development
failures**; only three support numbers reported in the paper. This file maps every run to its files
and states what each run is evidence for.

> **The single most important thing to know:** `workspace/runN/runfile.py` is the file left in the
> working directory when a run ends — the **last node the agent attempted**, not the node that
> produced the results. In Runs 5 and 7 it constructs `ImplicitMF()` and `ItemItem()` without their
> required arguments and raises `TypeError` before any experiment runs. The evaluated code is
> `runfile_iter9_bestnode.py` (Run 5) and `runfile_iter2_bestnode.py` (Run 7), archived under
> [`artifacts/`](artifacts/).

## Where things live

| Path | Contents |
|---|---|
| [`artifacts/run5-selected-node/`](artifacts/run5-selected-node/) | The node behind every Run 5 number in the paper. **Start here.** |
| [`artifacts/run7-selected-node/`](artifacts/run7-selected-node/) | The node behind every Run 7 number in the paper. |
| [`artifacts/VALIDATION_REPORT.md`](artifacts/VALIDATION_REPORT.md) | Recorded output of the completeness and preprocessing checks. |
| [`artifacts/RERUN_REPORT.md`](artifacts/RERUN_REPORT.md) | Recorded output of the re-execution, the independent metric recomputation, and the corrected-ItemKNN counterfactual. |
| `out/runN/` | Agent logs per run (`run_stdout.log`, sometimes `debug.log`, `save.pkl`). |
| `workspace/runN/` | Generated code (`runfile.py`, `runfile_iter<N>_bestnode.py`) and its output in `working/`. |
| `workspace/*.csv`, `*.dat`, `u.data` | Input datasets handed to the agent. |
| `scripts/` | Our own verification code — see the last section. |

## Runs that support results in the paper

| | Run 5 | Run 7 |
|---|---|---|
| Date | 2026-05-08 | 2026-07-16 |
| Archive | [`artifacts/run5-selected-node/`](artifacts/run5-selected-node/) | [`artifacts/run7-selected-node/`](artifacts/run7-selected-node/) |
| Evaluated node | tree-search iteration **9 of 10** | iteration **2 of 10**, node `dac1170c438449f5a248475720dd3267` |
| Reviewer score | 72/100, `is_satisfactory=False` | 72/100, `is_satisfactory=False` |
| Node code | `workspace/run5/runfile_iter9_bestnode.py` | `workspace/run7/runfile_iter2_bestnode.py` |
| Prompt | `prompt_run5_paperreplikation.txt` | same |
| Config | `config.toml` (`gpt-5.4`, 3 drafts, 10 iterations) | same |
| Log | `out/run5/run_stdout.log` | `out/run7/run_stdout.log`, `out/run7/debug.log` |
| Tree-search state | **none** — overwritten by Run 7 | `out/run7/save.pkl` |
| Per-seed results | `workspace/run5/working/seed_sensitivity_results.csv` | `workspace/run7/working/seed_sensitivity_results.csv` |
| Used for | Tables 2–6, Figure 2 | Tables 2 and 5, the replication paragraph |
| Methodological status | 6/9 combinations valid; the 3 ItemKNN combinations invalid | same |

Each archive directory carries its own `README.md` explaining every file, a `NODE.json` with
machine-readable provenance including code hashes, and a `reviewer_score.txt` with the complete
per-requirement feedback the paper quotes.

### Run 5 — + P5, correct Amazon file, batch-recommend hint · **primary result**

- **Prompt:** `prompt_run5_paperreplikation.txt` · **Config:** `config.toml`, model `gpt-5.4`
- **Selected node:** tree-search iteration 9 of 10, reviewer score 72/100, not satisfactory
- **Log:** `out/run5/run_stdout.log` — node code embedded at line 556, reviewer score at line 561,
  best-node fallback at line 609
- **Results:** `workspace/run5/working/seed_sensitivity_results.csv` (45 rows, 270 metric values),
  `seed_sensitivity_summary.csv`, `paired_stats.csv`, `seed_effect_summary.csv`, `run_manifest.json`
- **Supports:** Tables 3, 4 and 6, Figure 2, and the Run 5 rows of Tables 2 and 5
- **Caveat:** the three ItemKNN sub-experiments are complete but not methodologically validated

### Run 7 — repeat of Run 5 under identical settings · **replication**

- **Prompt and config:** identical to Run 5
- **Selected node:** iteration 2 of 10, node `dac1170c438449f5a248475720dd3267`, score 72/100
- **Log:** `out/run7/run_stdout.log`, `out/run7/debug.log`, `out/run7/save.pkl`
- **Results:** `workspace/run7/working/seed_sensitivity_results.csv` (45 rows, 270 metric values),
  `metric_summary_mean_std.csv`, `cv_summary.csv`, `paired_ttests_all_metrics.csv`
- **Supports:** the replication paragraph and the Run 7 rows of Tables 2 and 5
- **Caveats:** same ItemKNN defect as Run 5. Run 7 wrote its own metric implementation, independent
  of Run 5's. Its reviewer marked the implicit-feedback requirement *fulfilled*, unlike Run 5's.

### Run 6 — reduced prompt · **small-task result**

- **Prompt:** `prompt_run6_eigenerprompt.txt` (ALS only, MovieLens 100K, 3 seeds)
- **Selected node:** iteration 1, score 83.33/100 · **Code:** `workspace/run6/runfile_iter1_bestnode.py`
- **Log:** `out/run6/run_stdout.log` · **Results:** `workspace/run6/working/ml100k_implicitmf_results.csv`
- **Supports:** the small-task section. The only one of our runs that is methodologically sound
  throughout, because a single-algorithm prompt raises no feedback-mode configuration decision.

## Runs that document development failures

| Run | Date | What it shows | Usable output? |
|---|---|---|---|
| **1** | 2026-05-07 | Original code on Windows. Hung in tree-search iteration 4. Revealed the missing `matplotlib` dependency (P1) and the relative log path that breaks multiprocessing spawn (P2). Log: `out/run1/run_stdout.log`, `out/run1/debug.log`. | No. `workspace/run1/working/` holds a single intermediate `.npy`. |
| **2** | 2026-05-07 | With P1+P2, using `gpt-5.4-mini`. All 10 iterations ran, but 6 of 13 debugging iterations failed on hallucinated LensKit APIs. The run then crashed with `IndexError` because the "good node" list was empty (revealed P3). Log: `out/run2/run_stdout.log`, `out/run2/save.pkl`. | **Metrics exist but are degenerate.** See the note below. Evidence for the model observation and for P3; not a source of any reported value. |
| **3** | 2026-05-07 | With P3. Crashed during the first draft with `PermissionError: [WinError 5]` — `os.kill(pid, SIGINT)` is not valid on Windows (revealed P4). Zero tree-search iterations. | No. `workspace/run3/working/` is empty. |
| **4** | 2026-05-08 | With P4. Reached iteration 7, then hung on the output-queue deadlock (revealed P5). **Its Amazon input was the defective `VideoGames_old_nocolumn.csv`** — no header row, so the "rating" column is actually a timestamp and the `rating > 3` filter dropped none of its 231,780 rows. | **Partly.** Amazon results rest on corrupted input. MovieLens and Last.FM used the same data as Run 5, and this run configured ItemKNN **correctly** (`ItemItem(nnbrs=20, feedback='implicit')`, `workspace/run4/runfile.py:183`). Used **only** as the ItemKNN control. `workspace/run4/working/seed_metrics.csv`. |

**Why Run 2 counts as 0/9 methodologically validated despite writing 45 complete rows.** In
`workspace/run2/working/all_results.csv`, ALS and ItemKNN return **bit-identical** nDCG@10 in all
15 dataset–seed cells, and both are **exactly 0.0** on Amazon Video Games (ML100K
0.131/0.112/0.103/0.082/0.122; Last.FM 0.005/0.026/0.007/0.012/0.023). Two different algorithms
cannot agree to the last digit on 15 independent splits. The file is complete; the values are not
measurements.

**Run 4's ItemKNN control values** are five-seed means, computed from
`workspace/run4/working/seed_metrics.csv`: MovieLens 100K nDCG@10 = **0.2557** (ALS 0.2062),
Last.FM **0.1164**. Individual seeds differ — seed 1 alone gives 0.2458 and 0.1051 — so any
comparison against them has to state which is meant.

## Datasets

| File | Used by | Note |
|---|---|---|
| `workspace/u.data` | all runs | MovieLens 100K, 100,000 ratings. After `rating > 3` and 5-core: 54,413 interactions / 938 users / 1,008 items. |
| `workspace/VideoGames.csv` | Runs 5, 6, 7 | Amazon Video Games, 814,586 rows, header `user_id, parent_asin, rating, timestamp`. After `rating > 3` and 5-core: 533,133 / 63,383 / 19,020. **Correct file.** |
| `workspace/VideoGames_old_nocolumn.csv` | Run 4 | **Defective.** No header row, and the third column is a timestamp, not a rating. Kept only to document the defect. |
| `workspace/Video_Games.csv` | — | Byte-identical copy of `VideoGames.csv`. |
| `workspace/UserTaggedArtists-timestamps.dat` | all runs | Last.FM HetRec 2011. After deduplication and 5-core: 52,551 / 1,090 / 3,646. |

All three post-5-core counts were recomputed independently by `scripts/validate_results.py` from
the raw files and match the values recorded in `workspace/run5/working/run_manifest.json` exactly.

## Verification scripts

| Script | What it does | Needs LensKit? |
|---|---|---|
| [`scripts/validate_results.py`](scripts/validate_results.py) | 13 checks over Runs 5 and 7: 45 rows, 5 distinct seeds per combination, no missing or duplicate combinations, no NaN or non-finite values, 270 metric values in [0,1], recomputed means against Table 3, 5-core counts recomputed from the raw data, and the resolved ItemKNN configuration. Sets an exit code; `--json` writes a machine-readable report. | Only for the last check, which is skipped without it |
| [`scripts/rerun_selected_node.py`](scripts/rerun_selected_node.py) | Re-executes the evaluated Run-5 node's pipeline, compares against the reported per-seed values, recomputes the metrics with an independent implementation, checks the split for leakage and the candidate sets for seen items, and runs the corrected-ItemKNN counterfactual. | Yes |

Recorded output: [`artifacts/VALIDATION_REPORT.md`](artifacts/VALIDATION_REPORT.md) (13 of 13
checks pass) and [`artifacts/RERUN_REPORT.md`](artifacts/RERUN_REPORT.md). The machine-readable
report is [`artifacts/validation_report.json`](artifacts/validation_report.json). Raw per-cell
re-execution results are in [`scripts/rerun_results.csv`](scripts/rerun_results.csv).

## Patches

The five defects we fixed (P1–P5) are in `treesearch/`, `utils/`, `viz.py` and `pyproject.toml`;
see the [README](README.md) and Table 1 of the paper. A sixth (I1, a `UnicodeEncodeError` on
non-ASCII LLM output) was reported upstream and fixed there.
