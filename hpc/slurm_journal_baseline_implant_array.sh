#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-12:00:00
#SBATCH --job-name=cond_drift_base
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-wgan,diffusion}"
BASELINE_SUITE_DIR="${BASELINE_SUITE_DIR:-$PROJECT_ROOT/results/journal_baseline_implants_${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}}"
DATASET_SIZE="${DATASET_SIZE:--1}"
BATCH_SIZE="${BATCH_SIZE:--1}"
METRIC_EVAL_SIZE="${METRIC_EVAL_SIZE:-100000}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:--1}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:--1}"
WGAN_EPOCHS="${WGAN_EPOCHS:--1}"
DIFFUSION_LEARNING_RATE="${DIFFUSION_LEARNING_RATE:-0.0001}"
DIFFUSION_DDIM_STEPS="${DIFFUSION_DDIM_STEPS:-100}"
DIFFUSION_EVAL_BATCH_SIZE="${DIFFUSION_EVAL_BATCH_SIZE:--1}"
SKIP_METRIC_EVAL="${SKIP_METRIC_EVAL:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$BASELINE_SUITE_DIR"

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
echo "[slurm] baseline_suite_dir: $BASELINE_SUITE_DIR"
echo "[slurm] seed_start: $SEED_START"
echo "[slurm] num_seeds: $NUM_SEEDS"
echo "[slurm] channels: $CHANNELS"
echo "[slurm] variants: $VARIANTS"
echo "[slurm] dataset_size: $DATASET_SIZE"
echo "[slurm] batch_size: $BATCH_SIZE"
echo "[slurm] metric_eval_size: $METRIC_EVAL_SIZE"
echo "[slurm] diffusion_epochs: $DIFFUSION_EPOCHS"
echo "[slurm] wgan_epochs: $WGAN_EPOCHS"
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
    "$PYTHON_BIN" -u scripts/run_journal_baseline_implant_seed_channel.py
    --device cuda
    --seed "$CURRENT_SEED"
    --channel "$CURRENT_CHANNEL"
    --variants "$VARIANTS"
    --suite-dir "$BASELINE_SUITE_DIR"
    --dataset-size "$DATASET_SIZE"
    --batch-size "$BATCH_SIZE"
    --metric-eval-size "$METRIC_EVAL_SIZE"
    --swd-projections "$SWD_PROJECTIONS"
    --diffusion-epochs "$DIFFUSION_EPOCHS"
    --wgan-epochs "$WGAN_EPOCHS"
    --diffusion-learning-rate "$DIFFUSION_LEARNING_RATE"
    --diffusion-ddim-steps "$DIFFUSION_DDIM_STEPS"
    --diffusion-eval-batch-size "$DIFFUSION_EVAL_BATCH_SIZE"
  )
  if [[ "$SKIP_METRIC_EVAL" == "1" || "$SKIP_METRIC_EVAL" == "true" || "$SKIP_METRIC_EVAL" == "yes" ]]; then
    CMD+=(--skip-metric-eval)
  fi
  if [[ "$FORCE_RETRAIN" == "1" || "$FORCE_RETRAIN" == "true" || "$FORCE_RETRAIN" == "yes" ]]; then
    CMD+=(--force-retrain)
  fi

  "${CMD[@]}"
done
