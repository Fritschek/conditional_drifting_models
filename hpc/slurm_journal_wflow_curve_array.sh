#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=cond_drift_curve
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-10}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
CURVE_SUITE_DIR="${CURVE_SUITE_DIR:-$PROJECT_ROOT/results/journal_wflow_curves_${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}}"
AE_DATASET_SIZE="${AE_DATASET_SIZE:-1000000}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-500}"
AE_EPOCHS="${AE_EPOCHS:-10}"
AE_LEARNING_RATE="${AE_LEARNING_RATE:-0.001}"
EVAL_SIZE="${EVAL_SIZE:-100000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1000}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EBNO_VALUES="${EBNO_VALUES:-}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

if [[ -z "${WFLOW_SUITE_DIR:-}" ]]; then
  echo "WFLOW_SUITE_DIR must point to the completed journal W-Flow checkpoint suite." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$CURVE_SUITE_DIR"

IFS=',' read -r -a CHANNEL_ARRAY <<< "$CHANNELS"
CHANNEL_COUNT="${#CHANNEL_ARRAY[@]}"
if [[ "$CHANNEL_COUNT" -lt 1 ]]; then
  echo "CHANNELS must contain at least one channel" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((NUM_SEEDS * CHANNEL_COUNT))

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] wflow_suite_dir: $WFLOW_SUITE_DIR"
echo "[slurm] curve_suite_dir: $CURVE_SUITE_DIR"
echo "[slurm] seed_start: $SEED_START"
echo "[slurm] num_seeds: $NUM_SEEDS"
echo "[slurm] channels: $CHANNELS"
echo "[slurm] variants: $VARIANTS"
echo "[slurm] ebno_values: ${EBNO_VALUES:-channel-default}"
echo "[slurm] logical_task_count: $TASK_COUNT"
echo "[slurm] array_task_count: $ARRAY_TASK_COUNT"
echo "[slurm] job_id: ${SLURM_JOB_ID:-n/a}"
echo "[slurm] array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

echo "[slurm] GPU information"
nvidia-smi || true

for ((LOGICAL_TASK_ID = TASK_ID; LOGICAL_TASK_ID < TASK_COUNT; LOGICAL_TASK_ID += ARRAY_TASK_COUNT)); do
  SEED_OFFSET=$((LOGICAL_TASK_ID / CHANNEL_COUNT))
  CHANNEL_OFFSET=$((LOGICAL_TASK_ID % CHANNEL_COUNT))
  CURRENT_SEED=$((SEED_START + SEED_OFFSET))
  CURRENT_CHANNEL="${CHANNEL_ARRAY[$CHANNEL_OFFSET]}"

  echo "[slurm] running seed=$CURRENT_SEED channel=$CURRENT_CHANNEL logical_task_id=$LOGICAL_TASK_ID"

  CMD=(
    "$PYTHON_BIN" -u scripts/run_journal_wflow_curve_seed_channel.py
    --device cuda
    --seed "$CURRENT_SEED"
    --channel "$CURRENT_CHANNEL"
    --variants "$VARIANTS"
    --wflow-suite-dir "$WFLOW_SUITE_DIR"
    --suite-dir "$CURVE_SUITE_DIR"
    --ae-dataset-size "$AE_DATASET_SIZE"
    --ae-batch-size "$AE_BATCH_SIZE"
    --ae-epochs "$AE_EPOCHS"
    --ae-learning-rate "$AE_LEARNING_RATE"
    --eval-size "$EVAL_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --eval-every "$EVAL_EVERY"
  )
  if [[ -n "$EBNO_VALUES" ]]; then
    CMD+=(--ebno-values "$EBNO_VALUES")
  fi
  if [[ "$FORCE_RETRAIN" == "1" || "$FORCE_RETRAIN" == "true" || "$FORCE_RETRAIN" == "yes" ]]; then
    CMD+=(--force-retrain)
  fi

  "${CMD[@]}"
done
