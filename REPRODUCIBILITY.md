# Reproducibility guide

Two levels are supported. [STATUS.md](STATUS.md) lists what each one covers.

## Level 1 — figures and numbers from the released data

Requirements: Python 3.10+, the six packages in `requirements.txt`. No GPU, no
model weights, no network access after installation.

```bash
pip install -r requirements.txt
make validate     # python -m scripts.validate_release
make figures      # python -m scripts.make_figures
make web          # python -m scripts.export_web_data
```

- `make validate` checks every released file against
  `data/release_manifest.json` (SHA-256, columns, row counts), checks the
  stimulus design (balance, pairing, matched controls), and recomputes the
  headline numbers of the report from the released tables. It exits non-zero on
  any mismatch. Add `--verbose` to the Python command to list every check.
- `make figures` writes the report figures to `figures/` and the exploratory
  figures to `figures/supplementary/`. It reads nothing outside `data/`.
- `make web` rewrites `web_export/*.json`. It is deterministic.

The committed PNGs were rendered with a conda build of matplotlib 3.10.8
(FreeType 2.14) and are reproduced pixel for pixel by that build. The pip
wheel pinned in `requirements.txt` bundles an older FreeType and draws text
edges slightly differently; plotted data, layout and image size are the same.
Figures that contain a PCA projection can also differ in the last digits of
the projected coordinates across linear-algebra libraries.

## Level 2 — rerunning the models

### 1. Environment

```bash
pip install -r requirements-full.txt
```

`requirements-full.txt` pins the versions of the environment that produced the
released results (torch 2.11.0, transformers 4.51.0). Install the torch build
that matches your CUDA version.

### 2. Model weights

The eight checkpoints are listed with their Hugging Face revisions in
`configs/study.json`. The pipeline loads models by name and does not pin the
revision itself, so to match the release download those revisions explicitly,
for example:

```bash
export HF_HOME=/path/with/space/hf_cache
huggingface-cli download Qwen/Qwen2.5-7B --revision d149729398750b98c0af14eb82c78cfe92750796
```

Runtime switches (all optional, read in `utils/hpc.py`):

| Variable | Default | Meaning |
|---|---|---|
| `HPC_OFFLINE` | `1` | `1` loads only from the local cache; set `0` to allow downloads |
| `HF_HOME` | `./hf_cache` | Hugging Face cache directory |
| `OFFLOAD_DIR` | `./offload_dir` | scratch directory for `device_map` offloading |
| `HF_DEVICE_MAP` | `auto` | passed to `from_pretrained` when a GPU is available |

On a GPU, weights are loaded in bfloat16; on CPU in float32.

### 3. Pipeline

Run from the repository root. Each step writes below `results/` (git-ignored).

```bash
WORDS="bank bark bat crane spring match pitch"

# (a) Profiling forward passes: per-layer states of the 280 profiling sentences
#     at the homonym and at the final token, plus per-layer GDV.
#     -> results/activations/, results/activations_final/, results/<model>_gdv/
python run_h2.py --words $WORDS

# (b) GDV, nested H1 and held-out H2 from those caches. CPU only.
#     -> results/study/H1/, results/study/H2/, results/study/geometry_inference.csv
python -m analysis.recompute_geometry

# (c) Forward passes on the carriers, paired sentences and conflict items.
#     -> results/study/H0/, H3/, H4/, H5/
python run_study.py --hypotheses H0 H3 H4 H5 --words $WORDS

# (d) Post-hoc associations and the H5 architecture bootstrap. CPU only.
python -m analysis.analyze_cross_hypothesis_links

# (e) Optional exploratory extension (not in the report): layer selection on
#     L-condition sentences and H5 repeated at every candidate layer.
#     -> results/study/Q4/, results/study/H5_q4/
python run_q4_endpoint_analysis.py --words $WORDS --tolerance 0.05 --min-adequacy 0.6
python -m analysis.analyze_q4_impact

# (f) Optional sample trajectories (one encoder, one decoder, "bank").
python -m analysis.plot_h3_context_trajectory --models roberta qwen7b --words bank
python -m analysis.plot_h5_revision_trajectory --models roberta qwen7b --words bank
```

`scripts/slurm/run_full_study.example.slurm` wraps steps (a)–(d) for a Slurm
cluster.

### 4. From a rerun to the released tables

```bash
python -m scripts.build_release_data --results-dir results
make validate && make figures && make web
```

`build_release_data` needs the outputs of steps (a)–(f). It copies values as
text, verifies that the released tables reproduce every source file, and
rewrites `data/release_manifest.json`. After a rerun, `make validate` compares
your numbers with those printed in the report; small deviations are expected
(see below) and show up as failed headline checks.

### Compute and expected deviations

- Original hardware: one node with 4× NVIDIA A100. With cached weights the
  forward-pass jobs each took roughly 10–15 minutes.
- Disk: about 120 GB for the model cache, about 2 GB for the full per-layer
  profiling caches under `results/`.
- Hidden states depend on hardware, CUDA and library versions, and on bfloat16
  arithmetic, so margins can differ in the last digits and a near-zero margin
  can change sign. A full rerun on a clean machine has not been verified for
  this release.
- Bootstrap intervals use fixed seeds and reproduce exactly given the same
  inputs.

### Tests

```bash
make test        # python -m pytest -q tests   (needs requirements-full.txt)
```

Fourteen unit tests cover the margin and GDV computations, the H0 carrier
collapsing, the H4 transition summary and the H5 stimulus audit and prefix
construction. They do not download or run any model.
