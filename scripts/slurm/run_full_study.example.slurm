#!/usr/bin/env bash
# Example Slurm launcher for a full rerun (extraction + H0-H5 + figures).
# Copy, adjust the #SBATCH lines and the three paths below for your cluster.
# Site-specific launchers (*.slurm in the repository root) are git-ignored.
#SBATCH --job-name=llm_homonym_full_study
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR to the repository checkout}
PYTHON=${PYTHON:-python}                      # interpreter with requirements-full.txt
export HF_HOME=${HF_HOME:-$PROJECT_DIR/hf_cache}   # where model weights are (or will be) cached

cd "$PROJECT_DIR"
mkdir -p logs

# 1 = use only weights already in $HF_HOME (compute nodes without internet);
# 0 = allow downloads from the Hugging Face Hub.
export HPC_OFFLINE=${HPC_OFFLINE:-1}
export OFFLOAD_DIR=${OFFLOAD_DIR:-$PROJECT_DIR/offload_dir}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DEVICE_MAP=auto
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

WORDS="bank bark bat crane spring match pitch"

# 1. Profiling forward passes -> results/activations{,_final}/ and per-layer GDV.
"$PYTHON" run_h2.py --words $WORDS
"$PYTHON" -m analysis.validate_study_run --preflight
touch "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.start"

# 2. GDV, nested H1 and held-out H2 from the cached activations (CPU only).
"$PYTHON" -m analysis.recompute_geometry

# 3. Forward-pass analyses on the paired and context-conflict stimuli.
"$PYTHON" run_study.py --hypotheses H0 H3 H4 H5 --words $WORDS

# 4. Post-hoc associations and H5 architecture bootstrap (CPU only).
"$PYTHON" -m analysis.analyze_cross_hypothesis_links

# 5. Figures straight from results/ (the release instead uses `make figures`).
"$PYTHON" -m analysis.plot_geometry_audit
"$PYTHON" -m analysis.plot_geometry_audit --annotate-overlaps \
    --output results/study/figures/h2_gdv_geometric_example_overlap_annotated.svg
"$PYTHON" -m analysis.plot_semantic_layer_atlas
"$PYTHON" -m analysis.plot_prior_matrix
"$PYTHON" -m analysis.plot_h3_h4_audit
"$PYTHON" -m analysis.plot_report_h1_h5

"$PYTHON" -m analysis.validate_study_run --newer-than "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.start" --full

# 6. Optional: refresh the released tables from this run.
# "$PYTHON" -m scripts.build_release_data --results-dir results
