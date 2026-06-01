#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=cond_drift_wflow_fix
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHANNELS="${REPAIR_CHANNELS:-${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}}"
DATASET_SIZE="${DATASET_SIZE:--1}"
EVAL_SIZE="${EVAL_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:--1}"
DRIFTING_EPOCHS="${DRIFTING_EPOCHS:--1}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:--1}"
SINKHORN_EPSILON="${SINKHORN_EPSILON:-}"
SINKHORN_MIN_EPSILON="${SINKHORN_MIN_EPSILON:-0.001}"
SINKHORN_ITERATIONS="${SINKHORN_ITERATIONS:-10}"
FIBER_GENERATED_SAMPLES="${FIBER_GENERATED_SAMPLES:-4}"
FIBER_POSITIVE_SAMPLES="${FIBER_POSITIVE_SAMPLES:-4}"
FIBER_REFERENCE_SAMPLES="${FIBER_REFERENCE_SAMPLES:-4}"
ANCHOR_METRICS="${ANCHOR_METRICS:-1}"
ANCHOR_COUNT="${ANCHOR_COUNT:-128}"
ANCHOR_SAMPLES="${ANCHOR_SAMPLES:-64}"
ANCHOR_SWD_PROJECTIONS="${ANCHOR_SWD_PROJECTIONS:-64}"

if [[ -z "${SUITE_DIR:-}" ]]; then
  echo "SUITE_DIR must be set to the W-Flow suite being repaired." >&2
  exit 1
fi
if [[ -z "${REPAIR_VARIANT:-}" || -z "${REPAIR_SEED:-}" ]]; then
  echo "REPAIR_VARIANT and REPAIR_SEED must be set." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SUITE_DIR"

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] suite_dir: $SUITE_DIR"
echo "[slurm] variant: $REPAIR_VARIANT"
echo "[slurm] seed: $REPAIR_SEED"
echo "[slurm] channels: $CHANNELS"
echo "[slurm] job_id: ${SLURM_JOB_ID:-n/a}"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

echo "[slurm] GPU information"
nvidia-smi || true

CMD=(
  "$PYTHON_BIN" -u scripts/run_journal_wflow_task.py
  --device cuda
  --seed "$REPAIR_SEED"
  --variant "$REPAIR_VARIANT"
  --channels "$CHANNELS"
  --dataset-size "$DATASET_SIZE"
  --eval-size "$EVAL_SIZE"
  --batch-size "$BATCH_SIZE"
  --drifting-epochs "$DRIFTING_EPOCHS"
  --swd-projections "$SWD_PROJECTIONS"
  --sinkhorn-min-epsilon "$SINKHORN_MIN_EPSILON"
  --sinkhorn-iterations "$SINKHORN_ITERATIONS"
  --fiber-generated-samples "$FIBER_GENERATED_SAMPLES"
  --fiber-positive-samples "$FIBER_POSITIVE_SAMPLES"
  --fiber-reference-samples "$FIBER_REFERENCE_SAMPLES"
  --anchor-count "$ANCHOR_COUNT"
  --anchor-samples "$ANCHOR_SAMPLES"
  --anchor-swd-projections "$ANCHOR_SWD_PROJECTIONS"
  --suite-dir "$SUITE_DIR"
)

if [[ -n "$SINKHORN_EPSILON" ]]; then
  CMD+=(--sinkhorn-epsilon "$SINKHORN_EPSILON")
fi
if [[ "$ANCHOR_METRICS" == "1" || "$ANCHOR_METRICS" == "true" || "$ANCHOR_METRICS" == "yes" ]]; then
  CMD+=(--anchor-metrics)
fi

"${CMD[@]}"
