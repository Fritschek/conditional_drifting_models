#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-100}"
VARIANTS="${VARIANTS:-kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA}"
DATASET_SIZE="${DATASET_SIZE:--1}"
EVAL_SIZE="${EVAL_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:--1}"
DRIFTING_EPOCHS="${DRIFTING_EPOCHS:--1}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:--1}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_ROOT/results/journal_wflow_paper_hpc_${SUITE_TAG}}"
MAX_PARALLEL="${MAX_PARALLEL:-}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SUITE_DIR"

ARRAY_END=$((NUM_SEEDS - 1))
ARRAY_SPEC="0-${ARRAY_END}"
if [[ -n "$MAX_PARALLEL" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"
fi

export PROJECT_ROOT
export SEED_START
export NUM_SEEDS
export VARIANTS
export CHANNELS
export DATASET_SIZE
export EVAL_SIZE
export BATCH_SIZE
export DRIFTING_EPOCHS
export SWD_PROJECTIONS
export SUITE_DIR

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] suite_dir: $SUITE_DIR"
echo "[submit] seeds: $SEED_START..$((SEED_START + NUM_SEEDS - 1))"
echo "[submit] variants: $VARIANTS"
echo "[submit] channels: $CHANNELS"
echo "[submit] dataset_size: $DATASET_SIZE"
echo "[submit] eval_size: $EVAL_SIZE"
echo "[submit] batch_size: $BATCH_SIZE"
echo "[submit] drifting_epochs: $DRIFTING_EPOCHS"
echo "[submit] swd_projections: $SWD_PROJECTIONS"
echo "[submit] tasks: $NUM_SEEDS"
echo "[submit] array: $ARRAY_SPEC"

ARRAY_JOB_ID=$(sbatch --parsable --array="$ARRAY_SPEC" hpc/slurm_journal_wflow_seed_array.sh)
echo "[submit] array_job_id: $ARRAY_JOB_ID"
echo "[submit] aggregate after completion with:"
echo "export PROJECT_ROOT=\"$PROJECT_ROOT\""
echo "export SUITE_DIR=\"$SUITE_DIR\""
echo "export SEED_START=$SEED_START"
echo "export NUM_SEEDS=$NUM_SEEDS"
echo "export VARIANTS=\"$VARIANTS\""
echo "export CHANNELS=\"$CHANNELS\""
echo "sbatch --dependency=afterok:${ARRAY_JOB_ID} hpc/slurm_journal_wflow_aggregate.sh"
