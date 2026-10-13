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
| Website data | `make web` → `web_export/` | deterministic; byte-identical on re-run |

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
| Per-layer hidden states for all models and words (about 2 GB as HDF5) | too large for the repository; regenerate with `python run_h2.py` |
| The H1 point-cloud figures (`analysis/plot_h1_layer_grid.py`, `plot_centroid_progression.py`, `plot_centroid_separation*.py`) | they read the full per-layer hidden states |
| Sample trajectories for words or models other than *bank* in RoBERTa-large and Qwen2.5-7B | only these two single-layer snapshots are released |
| Re-deriving `data/processed/` from scratch | steps in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |

A full rerun on a clean machine has **not** been verified for this release.
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
- No DOI has been minted yet, and no archive of the full hidden states exists
  outside the author's cluster.
