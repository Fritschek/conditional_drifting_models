#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=cond_drift_wflow
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-100}"
VARIANTS="${VARIANTS:-kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_ROOT/results/journal_wflow_hpc_${SUITE_TAG}}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,OptFib}"
DATASET_SIZE="${DATASET_SIZE:-120000}"
EVAL_SIZE="${EVAL_SIZE:-100000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DRIFTING_EPOCHS="${DRIFTING_EPOCHS:-60}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:-128}"
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
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SUITE_DIR"

IFS=',' read -r -a VARIANT_ARRAY <<< "$VARIANTS"
VARIANT_COUNT="${#VARIANT_ARRAY[@]}"
if [[ "$VARIANT_COUNT" -lt 1 ]]; then
  echo "VARIANTS must contain at least one variant" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((NUM_SEEDS * VARIANT_COUNT))

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] suite_dir: $SUITE_DIR"
echo "[slurm] seed_start: $SEED_START"
echo "[slurm] num_seeds: $NUM_SEEDS"
echo "[slurm] logical_task_count: $TASK_COUNT"
echo "[slurm] array_task_count: $ARRAY_TASK_COUNT"
echo "[slurm] variants: $VARIANTS"
echo "[slurm] job_id: ${SLURM_JOB_ID:-n/a}"
echo "[slurm] array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

echo "[slurm] GPU information"
nvidia-smi || true

for ((LOGICAL_TASK_ID = TASK_ID; LOGICAL_TASK_ID < TASK_COUNT; LOGICAL_TASK_ID += ARRAY_TASK_COUNT)); do
  SEED_OFFSET=$((LOGICAL_TASK_ID / VARIANT_COUNT))
  VARIANT_OFFSET=$((LOGICAL_TASK_ID % VARIANT_COUNT))
  CURRENT_SEED=$((SEED_START + SEED_OFFSET))
  CURRENT_VARIANT="${VARIANT_ARRAY[$VARIANT_OFFSET]}"

  echo "[slurm] running seed=$CURRENT_SEED variant=$CURRENT_VARIANT logical_task_id=$LOGICAL_TASK_ID"

  CMD=(
    "$PYTHON_BIN" -u scripts/run_journal_wflow_task.py
    --device cuda
    --seed "$CURRENT_SEED"
    --variant "$CURRENT_VARIANT"
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
done
