# Changelog

## 1.0.0 — 2026-10-07

First public, reproducible release.

### Added
- `data/processed/`: 30 versioned result tables covering every reported number
  and figure (H0–H5 and the exploratory Q4 extension), per model and per
  homonym as well as aggregated, plus three small single-layer hidden-state
  examples. Values are text-identical to the pipeline output.
- `data/stimuli/`: the complete stimulus set as five documented CSV tables
  (homonyms and senses, profiling sentences, ambiguous carriers, paired
  context sentences, context-conflict items with coherent controls and the
  exact prefixes given to the models).
- `data/release_manifest.json` (sizes, SHA-256, row counts, columns) and
  `data/README.md` (data dictionary).
- `scripts/make_figures.py` (`make figures`): regenerates all published
  figures from the released tables only.
- `scripts/validate_release.py` (`make validate`): schema, row-count, checksum
  and design checks, and the report's headline values recomputed from the
  released tables.
- `scripts/export_web_data.py` (`make web`) and `web_export/`: deterministic
  JSON exports for interactive website figures, with their own README.
- `scripts/build_release_data.py`, `scripts/build_stimulus_tables.py`: rebuild
  the released tables from a pipeline run, with a round-trip check.
- `analysis/`: the plotting and post-hoc analysis scripts, which previously
  lived outside the repository.
- `configs/study.json`: models with the Hugging Face revisions used, homonyms
  and senses, analysis constants.
- `STATUS.md`, `LICENSE` (MIT, code), `LICENSE-DATA` (CC BY 4.0, data),
  `CITATION.cff`, `environment.yml`, pinned `requirements.txt` and
  `requirements-full.txt`, `Makefile`, an example Slurm launcher.
- `figures/`: the regenerated figures.

### Changed
- README and REPRODUCIBILITY rewritten for external readers; the two
  reproduction levels are separated and the README no longer refers to scripts
  that were not in the repository.
- Profiling sentences are read from `data/profiling_sentences.json` instead of
  a pandas pickle (same content, verified equal). The loader still accepts the
  old `.pkl`.
- `torch` and `transformers` are imported only when a model is loaded, so the
  figure, validation and export paths run without them. `find_target_span`
  moved to `utils/text.py` and is re-exported from `models`.
- Importing the pipeline no longer creates `hf_cache/` and `offload_dir/`; they
  are created when a model is loaded.
- `.gitignore` rewritten: released CSV/JSON/figures are tracked; results,
  caches, weights, logs and site-specific launchers stay ignored.

### Fixed
- `analysis/plot_geometry_audit.py`: the overlap-annotated PCA figure could
  label one group of coincident points twice, depending on floating-point
  noise. It now writes one label per group. Plotted data and reported values
  are unchanged.
- Removed a hard-coded cluster path from `analysis/validate_study_run.py`.

### Removed
- `data/synthetic_data_h2.pkl` (replaced by the JSON file) and tracked
  `.DS_Store` files.

No reported result was recomputed or altered for this release.
