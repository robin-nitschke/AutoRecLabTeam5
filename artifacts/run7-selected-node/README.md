# Run 7 — selected node

Run 7 repeats Run 5 with an identical prompt, configuration and backend model. This directory
archives the node whose output produced the Run 7 numbers reported in the paper.

All files are **copies**; the originals remain at their paths in `out/run7/` and `workspace/run7/`.

## Which node this is

| | |
|---|---|
| Run | 7, executed 2026-07-16 |
| Tree-search iteration | **2 of 10** |
| **Node ID** | **`dac1170c438449f5a248475720dd3267`** |
| Parent draft node | `9c2fbb12e3134cfa84be0336f72b80ca` |
| Reviewer score | **72 / 100**, `is_satisfactory=False` |
| Why this node | It is the only node in Run 7 whose execution succeeded. The three draft nodes and iterations 1 and 3–10 were all flagged buggy, so AutoRecLab fell back to "Found no satisfactory node; Using best node instead" (`run_stdout.log:565`). |

### How the node ID was recovered

`pickle.load` on `save.pkl` returns only the **three draft root nodes** — `Scheduler.save()` pickles
`self._draft_nodes` ([treesearch/search.py:126-129](../../treesearch/search.py#L126-L129)). The
remaining nodes are reachable as their descendants:

```python
import pickle
from anytree import PreOrderIter

roots = pickle.load(open('save.pkl', 'rb'))          # 3 draft roots
nodes = [n for r in roots for n in PreOrderIter(r)]  # 13 nodes
good = [n for n in nodes if not n.is_buggy]          # exactly one
print(good[0].id)                                    # dac1170c438449f5a248475720dd3267
```

The 13 nodes correspond exactly to 3 drafts plus 10 tree-search iterations. Node
`dac1170c…`'s `code` attribute is byte-identical to `selected_node_code.py` (both 10,132
characters, SHA-256 `92f49bf1a178736617309d72be12a7d0`).

> **Do not cite `workspace/run7/runfile.py` as the source of these numbers.** As in Run 5, that file
> is a different, later node that constructs `ImplicitMF()` and `ItemItem()` with plain defaults.

## Methodological status of this node

| Algorithm | Constructor in the node | Status |
|---|---|---|
| ALS | `Recommender.adapt(ImplicitMF(features=50, iterations=15, reg=0.01, weight=40, method='cg'))` | valid — configured for implicit feedback |
| Pop | `Recommender.adapt(Popular())` | valid |
| ItemKNN | `Recommender.adapt(ItemItem(nnbrs=20, min_nbrs=1, min_sim=1.0e-6, center=False))` | **invalid** |

The ItemKNN configuration carries the **same defect as Run 5**: `feedback` is left at its
`'explicit'` default, so `aggregate` stays `'weighted-average'` and `use_ratings` stays `True`, and
every unary interaction scores exactly 1.0. Three of the nine sub-experiments are therefore
technically completed but scientifically invalid. See
[../RERUN_REPORT.md](../RERUN_REPORT.md).

That the same defect appears independently in Run 5 (iteration 9) and Run 7 (iteration 2) is itself
a finding: it is not a one-off.

**The reviewer caught it in Run 5 and missed it in Run 7.** Run 5's requirement 13 states that
ItemKNN "is not explicitly configured in an implicit-feedback mode as required/supported in LensKit
0.14.4". Run 7's requirement 10 — "Train each algorithm separately on each dataset/seed training
split using implicit-feedback data in the format expected by LensKit 0.14.4" — is marked
**Fulfilled**, and no requirement in Run 7's feedback mentions the feedback mode at all. The
constructor is character-for-character the same in both nodes (`reviewer_score.txt` in this
directory and in `../run5-selected-node/`). The reviewer is therefore not reliably wrong or
reliably right about this defect; it is inconsistent between two runs of the same prompt.

## Relationship to Run 5

Run 7 is a **repetition of the AutoRecLab pipeline**, not a re-execution of Run 5's generated code.
The agent produced different code that happens to share the ItemKNN defect. Run 7 therefore provides
evidence about the *system's* run-to-run behaviour, but does **not** by itself raise the evidence
level of Run 5's numbers.

## Files

| File | What it is |
|---|---|
| `selected_node_code.py` | The evaluated node's Python code, verbatim. Copy of `workspace/run7/runfile_iter2_bestnode.py`. |
| `NODE.json` | Machine-readable metadata: node ID, iteration, score, hashes, algorithm configuration, environment. |
| `save.pkl` | Tree-search state, from which the node ID above was recovered. Copy of `out/run7/save.pkl`. |
| `run_stdout.log` | Complete AutoRecLab console log of Run 7, including the reviewer score (line 267) and the final summary. Copy of `out/run7/run_stdout.log`. |
| `code_requirements.json` | The requirements the Planner derived from the prompt and against which the Reviewer scored the node. |
| `reviewer_score.txt` | **Reviewer score and full per-requirement feedback**, extracted from the `NodeScore` record at `run_stdout.log:267`. This is where requirement 10 is marked fulfilled despite the ItemKNN defect. |
| `pyproject.toml` | Dependency pins in force for this run, including `lenskit==0.14.4`. |
| `agent_debug.log` | AutoRecLab's own debug log for the run, alongside the console log. Copy of `out/run7/debug.log`. |
| `prompt.txt` | The natural-language prompt — identical to Run 5's. Copy of `prompt_run5_paperreplikation.txt`. |
| `config.toml` | Tree-search and model configuration in force for this run — identical to Run 5's. |
| `per_seed_results.csv` | **Per-seed result file** — 45 rows, one per dataset × algorithm × seed, with Precision and nDCG at cut-offs 1, 5 and 10. |
| `aggregated_results.csv` | **Aggregated result table** — mean and standard deviation per dataset × algorithm × metric. |
| `cv_summary.csv` | Coefficients of variation across the five seeds. |
| `statistical_output.csv` | **Statistical output** — paired *t*-tests across all metrics. Exploratory: five seed-matched pairs. |

## Environment

AutoRecLab Alpha (initial release, `main` branch), Team 5 fork with patches P1–P5 applied on top of
upstream commit `34444d3cc77127eb73075a44980056c11b566c12`. LensKit 0.14.4, Python 3.11, Windows 11,
backend model `gpt-5.4`. Identical to Run 5.

## Reproducing

See [../../REPRODUCING.md](../../REPRODUCING.md).
