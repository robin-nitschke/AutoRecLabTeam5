# Reproducing our results

There are two things you may want to reproduce, and they cost very different amounts of time and
money:

1. **The verification of our reported numbers** — free, offline, a few minutes. Start here.
2. **A full AutoRecLab run** — one to three hours and roughly \$0.5–1 of OpenAI API credit, and the
   agent will not produce the same code twice.

## Environment

| | |
|---|---|
| AutoRecLab | Alpha (initial release, `main` branch), upstream [ISG-Siegen/AutoRecLab](https://github.com/ISG-Siegen/AutoRecLab) |
| Baseline commit | `34444d3cc77127eb73075a44980056c11b566c12` ("chore: bump version to 1.0.0", 2026-01-30), with Team 5 patches P1–P5 applied on top |
| LensKit | **0.14.4** (pinned; the API changed substantially in 1.x and the generated code will not run on it) |
| Python | 3.11 (we used 3.11.9) |
| OS | Windows 11. Patches P2, P4 and P5 are Windows-specific; on Linux/macOS they are inert. |
| Hardware | AMD Ryzen 7 9800X3D (8C/16T), 32 GB RAM |
| Backend model | `gpt-5.4` (`config.toml`) |

## Installation

```bash
git clone https://github.com/robin-nitschke/AutoRecLabTeam5
cd AutoRecLabTeam5
uv sync                     # or: pip install -e .
```

`uv sync` installs from `uv.lock`, which pins LensKit to 0.14.4. If you install with pip, make sure
the pin holds — this is the single most common way to fail to reproduce anything here.

## Required data files

All three live in `workspace/` and are already in the repository:

| File | Dataset | Source |
|---|---|---|
| `u.data` | MovieLens 100K | GroupLens ML-100K |
| `VideoGames.csv` | Amazon Video Games reviews | header `user_id, parent_asin, rating, timestamp` |
| `UserTaggedArtists-timestamps.dat` | Last.FM HetRec 2011 | user–artist tag assignments |

Do **not** use `VideoGames_old_nocolumn.csv`. It has no header row, so its third column reads as a
rating when it is actually a timestamp; the `rating > 3` filter then drops nothing. It is kept only
to document the defect that affected Run 4.

## 1. Verify our reported numbers (recommended)

### Completeness and preprocessing checks — no LensKit needed

```bash
python scripts/validate_results.py
```

Runs 13 checks across Runs 5 and 7 and exits non-zero if any fails. Expected output:

```
13 of 13 checks passed.
```

Add `--json artifacts/validation_report.json` for a machine-readable report.

It verifies that each run has 45 algorithm–dataset–seed rows with five distinct seeds per
combination, no missing or duplicated combinations, no NaN or non-finite values, 270 metric values
all within [0,1], that the means and coefficients of variation recomputed from the per-seed file
match Table 3 of the paper, and that the 5-core interaction counts recomputed from the raw data
match `workspace/run5/working/run_manifest.json`:

```
ml100k             recomputed  54,413 /    938 /  1,008
amazon_videogames  recomputed 533,133 / 63,383 / 19,020
lastfm             recomputed  52,551 /  1,090 /  3,646
```

The last check is the only one that needs LensKit, and it is skipped without it. It instantiates
the archived `ItemItem` constructor and reads back the settings LensKit resolves it to:

```
as archived         : {'center': False, 'aggregate': 'weighted-average', 'use_ratings': True}
feedback="implicit" : {'center': False, 'aggregate': 'sum',              'use_ratings': False}
```

It **passes** when those differ, because the misconfiguration is what the paper reports about the
archived code. A failure would mean the archived node is not the one we describe.

The full recorded output is in [`artifacts/VALIDATION_REPORT.md`](artifacts/VALIDATION_REPORT.md).

### Re-execute the evaluated node — needs LensKit

```bash
python scripts/rerun_selected_node.py --datasets ml100k lastfm
```

Takes about two minutes. This re-runs the pipeline of the Run-5 node for all five seeds, compares
the new metrics with the reported ones, recomputes them with an independent implementation, checks
the split for train/test leakage and the candidate sets for items the user has already seen, and
additionally measures ItemKNN under the implicit-feedback configuration LensKit documents.

The script carries the node's pipeline functions as a transcription rather than importing
`selected_node_code.py`, so that the loop around them can run per dataset and per seed and write
one CSV row per cell. Section 0 of
[`artifacts/RERUN_REPORT.md`](artifacts/RERUN_REPORT.md) shows the abstract-syntax-tree comparison
proving the transcription is identical to the node, and gives the snippet to re-check it.

What you should see:

- **ItemKNN and Most-Popular reproduce exactly on MovieLens 100K and Last.FM** — zero difference
  against the reported values. Amazon ItemKNN does *not*, and that is itself a finding: with every
  score identical, the ranking is decided by set-iteration order, which depends on Python's
  per-process string hash seed for Amazon's string item IDs.
- **ALS does not.** The generated code passes no `rng_spec` to `ImplicitMF`, so its factor matrices
  are initialised unseeded. Per-seed values move by up to 7.6%; five-seed means stay within 1.7%.
- **`distinct_scores` is 1 for every ItemKNN cell.** This is the misconfiguration: with
  `feedback` left at its `'explicit'` default, the weighted-average aggregate over unary data gives
  every item a score of exactly 1.0.
- **`ItemKNN_implicit` scores far higher** — nDCG@10 rises 5.8× on MovieLens 100K and 24.8× on
  Last.FM.
- **`train_test_overlap` and `candidate_leakage_probe` are 0 in every cell** — no leakage between
  training and test data, and no candidate list contains an item the user already interacted with.

Amazon Video Games is much heavier (63,383 evaluation users × ~19,000 candidates each, run
single-process because the node's candidate function is a lambda and cannot be pickled):

```bash
python scripts/rerun_selected_node.py --datasets amazon_videogames --seeds 1
```

Results are appended to `scripts/rerun_results.csv` after every cell, so a partial run is still
usable. The full recorded output is in [`artifacts/RERUN_REPORT.md`](artifacts/RERUN_REPORT.md).

## 2. Run AutoRecLab end to end

```bash
export OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY = "sk-..."
python main.py
```

At the prompt, paste the contents of `prompt_run5_paperreplikation.txt` and then type `!start`.
`config.toml` controls the run:

```toml
[treesearch]
num_draft_nodes = 3
max_iterations  = 10
debug_prob      = 0.3
epsilon         = 0.3

[exec]
timeout   = 3600
workspace = "./workspace"

[agent.code]
model      = "gpt-5.4"
model_temp = 1.0
```

Expect one to three hours and roughly \$0.5–1 of API credit. Output lands in `out/` (console log,
`debug.log`, `save.pkl`, `code_requirements.json`) and `workspace/working/` (generated results).

### What to expect, and what to watch for

- **The agent will not generate the same code you see in `artifacts/`.** Runs 5 and 7 used an
  identical prompt and configuration and produced different code. Treat any single run as one draw.
- **Check the ItemKNN constructor before believing any ItemKNN number.** In both of our paper-scale
  runs the agent wrote `ItemItem(..., center=False)` and left `feedback` at `'explicit'`, which
  silently makes every score identical. The correct form is
  `ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, feedback='implicit')`. A quick check on the
  generated results: if a run's recommendation scores have only one distinct value, the model
  contributed nothing to the ranking.
- **No node is likely to be marked satisfactory.** Both of our paper-scale runs ended with "Found no
  satisfactory node; Using best node instead", and the fallback summary reports the node's numbers
  without mentioning its score or its unmet requirements. Read
  `out/run_stdout.log` for the `NodeScore(...)` line before using any result.
- **`out/save.pkl` is overwritten by every run.** Copy it out if you want to keep node identities.
  It contains only the draft roots; the rest of the tree is reachable through
  `anytree.PreOrderIter`.
- **`workspace/working/runfile.py` is the last node written, not necessarily the evaluated one.**
  Compare against the code embedded in the log or in `save.pkl`.

## Without our patches

On native Windows the unpatched Alpha release will not complete a run: it fails on a missing
`matplotlib` dependency, a relative log path that breaks multiprocessing spawn, an `IndexError`
when no node is good, a `PermissionError` from `os.kill(pid, SIGINT)`, and an output-queue deadlock
after the first execution timeout. See Table 1 of the paper and the repository
[README](README.md) for the individual fixes.
