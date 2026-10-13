# Release status

Release 1.0.0. This file states what can be reproduced from the repository
alone and what needs a model rerun.

## Reproducible from the released data (no models, no GPU)

Verified in a fresh Python 3.10 virtual environment containing only
`requirements.txt` (no torch, no transformers, no model cache): 278 of 278
validation checks pass, all figures are written, and the web export is
byte-identical to the committed one.

| What | How | Check |
|---|---|---|
| All 8 figures of the report | `make figures` → `figures/` | with the matplotlib build used for the report, 7 are pixel-identical to the originals; `h2_gdv_geometric_example_overlap_annotated` differs only in its overlap labels (see below) |
| 3 companion figures from the report's figure set (`h2_gdv_geometric_example`, `h2_layer_curves`, `h4_conditional_transitions`) | same | pixel-identical; the PCA panel differs by at most 2/255 in isolated pixels |
| 8 exploratory figures (Q4 curves and layer shift; sample trajectories for *bank*) | same → `figures/supplementary/` | Q4 figures pixel-identical; trajectory figures differ by at most 2/255 in isolated pixels |
| The headline numbers in the report's abstract, text and tables | `make validate` | 73 values recomputed and compared at printed precision |
| Stimulus design (balance, pairing, matched controls) | `make validate` | |
| Website data derived from tables (6 files) | `make web` → `web_export/` | deterministic; byte-identical on re-run |
| Structure and consistency of the two display-geometry exports | `make validate` | coordinates hash to the recorded values; density grids recompute exactly; full-space values equal the released tables |

Pixel identity depends on the matplotlib build. The originals and the
committed `figures/` were rendered with a conda build of matplotlib 3.10.8
(FreeType 2.14). The pip wheel of the same version bundles FreeType 2.6.1 and
draws text edges slightly differently (1–3% of pixels, all on glyph outlines);
data marks, layout and image size are the same.

One deliberate difference from the report: in the annotated geometry figure
the original code could label one overlap group twice, depending on
last-digit noise in the PCA coordinates. The label logic now emits one label
per group. Point positions, GDV and adequacy values are unchanged.

## Requires rerunning the models

| What | Why |
|---|---|
| Any new model, homonym, sentence or layer | hidden states are not released |
| Rebuilding `web_export/geometry_by_layer.json` | projects the per-layer hidden states of all models; `python -m scripts.build_geometry_exports` |
| Rebuilding `web_export/garden_path_landscapes.json` | projects sentinel states that need forward passes; `python -m scripts.extract_h5_matched_states`, then the command above |
| Per-layer hidden states for all models and words (about 2 GB as HDF5) | too large for the repository; regenerate with `python run_h2.py` |
| The H1 point-cloud figures (`analysis/plot_h1_layer_grid.py`, `plot_centroid_progression.py`, `plot_centroid_separation*.py`) | they read the full per-layer hidden states |
| Sample trajectories for words or models other than *bank* in RoBERTa-large and Qwen2.5-7B | only these two single-layer snapshots are released |
| Re-deriving `data/processed/` from scratch | steps in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |

The two display-geometry exports were built on the author's cluster from the
original hidden states. For the garden-path export one forward-pass job was run
on all eight models (4× A100, about 8 minutes) to read the coherent control
sentences at all three stages, which the original H5 run had read only at the
resolver. The same job repeated the published H5 inputs: all 2,352
conflicting-path margins agree with `h5_sentence_level.csv` to within 0.0001.
The coherent-control paths come from a separate forward pass and differ slightly
from the released control margin at the resolver stage (see "Batch dependence
of hidden states" below).

Apart from that job, a full rerun on a clean machine has **not** been verified
for this release.
The pipeline code is the code that produced the released results, with import
paths updated for the new layout and the model stack imported lazily; the unit
tests pass, but no forward pass was re-executed while preparing the release.
Reruns are expected to match the released values closely, not bit for bit.

## Audit: where things are

| Kind | Location | Released |
|---|---|---|
| Stimuli and sentence conditions | `data/profiling_sentences.json`, `data/paired_sentences.json`, `data/garden_path_sentences.json`; flat tables in `data/stimuli/` | yes |
| Model and configuration information | `utils/model_registry.py` (read by the code), `configs/study.json` (models with pinned revisions, words, constants) | yes |
| Processed results behind the findings | `data/processed/*.csv` (30 tables), built from `results/` by `scripts/build_release_data.py` | yes |
| Plotting and analysis code | `analysis/` (previously kept outside the repository), `scripts/` | yes |
| Pipeline code | `run_study.py`, `run_h2.py`, `run_q4_endpoint_analysis.py`, `hypotheses/`, `experiments/`, `models/`, `utils/` | yes |
| Full per-layer hidden states | `results/activations*/` on the author's cluster | no; three single-layer examples in `data/processed/example_states/` (2.6 MB) |
| Model weights | Hugging Face Hub | no; not redistributable here |
| Per-model GDV dashboards (`*_gdv.pkl`), PCA plot dumps, run logs, Slurm launchers with site paths, superseded result backups, working notebooks, report drafts | author's cluster | no; intermediate or site-specific |

## Known gaps

- The pipeline does not pin model revisions when loading; `configs/study.json`
  records the revisions that were in the author's cache.
- `transformers` 4.51.0 is the version installed in the environment that
  produced the results. Launch scripts also put an optional local development
  checkout of `transformers` on the path; whether one was active for the final
  runs could not be established.
- The released H5-by-layer table omits four free-text columns that repeat the
  stimulus text for every layer; they are recoverable by `sentence_id`.
- `data/synthetic/synthetic_datageneration.py` and
  `data/inspect_and_build_dataset.ipynb` are historical dataset-construction
  helpers. They refer to earlier file names and are not part of either
  reproduction path.
- Hidden states depend slightly on which texts share a batch (next section).
- `geometry_by_layer.json` holds within-layer standardised display coordinates
  in one shared PCA basis (each layer centred and scaled before the shared
  PCA; see `web_export/README.md`). Absolute cloud size is not comparable
  between layers.
- No DOI has been minted yet, and no archive of the full hidden states exists
  outside the author's cluster.

## Batch dependence of hidden states

A hidden state read by the pipeline depends slightly on which other texts are
in the same batch. This was found while building
`web_export/garden_path_landscapes.json` and was isolated with
`scripts/diagnose_batch_dependence.py` (all eight models, the 14 *bank* control
texts, H5 layer, NVIDIA A100, torch 2.11.0).

| Factor | Test | Result |
|---|---|---|
| Evaluation mode | `model.training` | `False` for all models; dropout is off |
| GPU nondeterminism | identical batches run twice | states identical, difference exactly 0, for all models |
| Padding and attention mask | float32 weights, different batch neighbours or no batching | margins change by at most 0.000002; padding is masked correctly |
| Precision | pipeline setting (bfloat16 weights and autocast), different batch neighbours | margins change by 0.001 to 0.008 depending on the model; states by up to 4% in norm |
| Own padding | bfloat16, rows that have no padding in either reading | they change as well |

Conclusion: the cause is bfloat16 arithmetic. With about three significant
digits, the result of a forward pass depends on the shape of the batch it is
computed in. It is deterministic for a fixed sequence of batches, which is why
repeating the published H5 inputs reproduces the released table, and it
disappears in float32. It is not a masking error and not run-to-run noise.

Consequences:

- All released margins are exactly reproducible only with the same texts in the
  same batches. A rerun that batches differently will differ in the third
  decimal, and margins very close to zero can change sign.
- In `garden_path_landscapes.json` every path comes from one forward-pass
  group and is never assembled from two. The conflicting paths repeat the
  published inputs and match the released table. The coherent-control paths
  were read in their own group, because the published run read the control at
  the resolver stage only. Over the 784 items their resolver margin differs
  from `matched_control_correct_margin_norm` by 0.0005 on average and by at
  most 0.027; one lies on the other side of the boundary. Each control
  path carries the released value (`released_table_resolver_margin`) and a flag
  (`resolver_side_differs_from_released_table`). This is a reproducibility
  discrepancy of the display export, not a change to any released table.
- Differences between two paths that are smaller than about 0.01 should not be
  interpreted.
- A float32 rerun of the forward-pass analyses would remove the dependence. It
  has not been done.
