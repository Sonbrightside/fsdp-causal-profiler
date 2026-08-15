#!/bin/bash
# GapScope census run — Slurm submission script.
# Usage: edit the CONFIG block, then:  sbatch experiments/submit_fsdp.sbatch

#SBATCH -p public
#SBATCH -q public
#SBATCH --job-name=gapscope_census
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=census_%j.log
# Add your allocation:  #SBATCH -A <your_account>

set -euo pipefail   # fail fast: any step that dies leaves its cause at the top of the log

# ── CONFIG (edit these) ─────────────────────────────────────────
REPO_ROOT=${GAPSCOPE_ROOT:-$HOME/gapscope}     # path to this repo
PYENV=$HOME/.conda/envs/fsdp312                # conda env with torch >= 2.9
CUDA_MODULE=cuda-12.8.1-gcc-12.1.0             # site CUDA module providing nvcc + CUPTI
# ────────────────────────────────────────────────────────────────

PYTHON=$PYENV/bin/python
TORCHRUN=$PYENV/bin/torchrun
TRACER_SO=$REPO_ROOT/tracer/libcupti_trace.so

# Environment: purge + explicit load blocks module drift between runs
module purge
module load $CUDA_MODULE
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export NVTX_INJECTION64_PATH=$TRACER_SO   # CUPTI NVTX domain capture

# Preflight: fail here, not after waiting in the queue on a ghost path
ls -la "$TRACER_SO"
$PYTHON -c "import torch; print('ENV CHECK:', torch.__version__, torch.version.cuda)"

# Trace output on node-local /tmp: avoids SQLite WAL over NFS
export CUPTI_TRACE_DIR=/tmp/census_$SLURM_JOB_ID
mkdir -p "$CUPTI_TRACE_DIR"

cd "$REPO_ROOT"

# LD_PRELOAD scoped to this one command only — never export it shell-wide
LD_PRELOAD=$TRACER_SO \
  $TORCHRUN --nproc_per_node=2 instrumentation/census_run.py

# Checkpoint WAL and collect traces
for f in "$CUPTI_TRACE_DIR"/*.sqlite; do
  sqlite3 "$f" "PRAGMA wal_checkpoint(TRUNCATE);"
done

DEST=$REPO_ROOT/traces/job_$SLURM_JOB_ID
mkdir -p "$DEST"
cp "$CUPTI_TRACE_DIR"/*.sqlite* "$DEST/"
echo "TRACES SAVED TO: $DEST"
ls -la "$DEST"