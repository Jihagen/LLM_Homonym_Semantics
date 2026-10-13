# Data in this repository

| Location | What | Documented in |
|---|---|---|
| `data/stimuli/`, `data/*.json` | the complete stimulus set | [data/README.md](data/README.md) |
| `data/processed/` | result tables behind every reported number and figure; three small single-layer hidden-state examples | [data/README.md](data/README.md) |
| `web_export/` | JSON files for the research website | [web_export/README.md](web_export/README.md) |
| `data/release_manifest.json` | every file above with size and SHA-256; row counts and columns for tables; item counts and source hashes for the display-geometry exports | |

## Analysis results and display coordinates

- **Analysis results** are the values in `data/processed/`: margins, fractions
  adequate and GDV, all computed in the full hidden space of the model.
- **Display coordinates** are in `web_export/geometry_by_layer.json` and
  `web_export/garden_path_landscapes.json`: three-dimensional PCA projections of
  hidden states, and kernel-density grids of those projections, for plotting.
  They keep only part of the variance, are marked `preliminary`, and are not
  used by any analysis. The all-layer coordinates are within-layer standardised
  display coordinates in one shared PCA basis: they show cluster separation
  across depth, and absolute cloud size is not comparable between layers.
  Full-space GDV, adequacy and margins remain the analysis metrics. The projection conventions (shared basis across layers,
  normalisation, solver, rounding, density bandwidth) are specified in
  [web_export/README.md](web_export/README.md).

## What is not published

Full hidden states (about 2 GB of per-layer profiling states, plus the sentinel
states behind the garden-path export), model weights and caches. Each
display-geometry cell records a SHA-256 of the hidden states it was projected
from, so a regenerated set of states can be checked against the release.
[STATUS.md](STATUS.md) lists what can be reproduced without them.

Licence: CC BY 4.0, see [LICENSE-DATA](LICENSE-DATA).
