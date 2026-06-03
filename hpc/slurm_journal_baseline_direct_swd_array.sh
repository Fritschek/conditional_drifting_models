#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=cond_drift_base_swd
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE_CHECKPOINT_SUITE_DIR="${BASELINE_CHECKPOINT_SUITE_DIR:?BASELINE_CHECKPOINT_SUITE_DIR must point to the WGAN/diffusion checkpoint suite.}"
DIRECT_SWD_SUITE_DIR="${DIRECT_SWD_SUITE_DIR:-$PROJECT_ROOT/results/journal_baseline_direct_swd_${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-TDL}"
VARIANTS="${VARIANTS:-wgan,ddpm,ddim100}"
EVAL_SIZE="${EVAL_SIZE:--1}"
BATCH_SIZE="${BATCH_SIZE:--1}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:--1}"
DIFFUSION_DDIM_STEPS="${DIFFUSION_DDIM_STEPS:-100}"
WGAN_NORMALIZE_CONDITION="${WGAN_NORMALIZE_CONDITION:-1}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$DIRECT_SWD_SUITE_DIR"

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
echo "[slurm] checkpoint_suite_dir: $BASELINE_CHECKPOINT_SUITE_DIR"
echo "[slurm] direct_swd_suite_dir: $DIRECT_SWD_SUITE_DIR"
echo "[slurm] seed_start: $SEED_START"
echo "[slurm] num_seeds: $NUM_SEEDS"
echo "[slurm] channels: $CHANNELS"
echo "[slurm] variants: $VARIANTS"
echo "[slurm] eval_size: $EVAL_SIZE"
echo "[slurm] batch_size: $BATCH_SIZE"
echo "[slurm] swd_projections: $SWD_PROJECTIONS"
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
    "$PYTHON_BIN" -u scripts/evaluate_journal_baseline_direct_swd.py
    --device cuda
    --suite-dir "$BASELINE_CHECKPOINT_SUITE_DIR"
    --out-dir "$DIRECT_SWD_SUITE_DIR"
    --channels "$CURRENT_CHANNEL"
    --seeds "$CURRENT_SEED"
    --variants "$VARIANTS"
    --eval-size "$EVAL_SIZE"
    --batch-size "$BATCH_SIZE"
    --swd-projections "$SWD_PROJECTIONS"
    --diffusion-ddim-steps "$DIFFUSION_DDIM_STEPS"
    --skip-summary
  )
  if [[ "$WGAN_NORMALIZE_CONDITION" == "0" || "$WGAN_NORMALIZE_CONDITION" == "false" || "$WGAN_NORMALIZE_CONDITION" == "no" ]]; then
    CMD+=(--no-wgan-normalize-condition)
  fi

  TQDM_DISABLE=1 "${CMD[@]}"
done
