#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=cond_drift_wall
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-analytic,fiber_sinkhorn,wgan,diffusion_ddim100}"
WFLOW_SUITE_DIR="${WFLOW_SUITE_DIR:-}"
WFLOW_SUITE_DIR_MAP="${WFLOW_SUITE_DIR_MAP:-}"
BASELINE_SUITE_DIR="${BASELINE_SUITE_DIR:-}"
WALLCLOCK_SUITE_DIR="${WALLCLOCK_SUITE_DIR:-$PROJECT_ROOT/results/journal_equal_wallclock_symbolic_${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}}"
TRAIN_SECONDS="${TRAIN_SECONDS:-1800}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-500}"
AE_LEARNING_RATE="${AE_LEARNING_RATE:-0.001}"
EVAL_SIZE="${EVAL_SIZE:-100000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1000}"
DIFFUSION_DDIM_STEPS="${DIFFUSION_DDIM_STEPS:-100}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
LOG_EVERY="${LOG_EVERY:-500}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

if [[ -z "$WFLOW_SUITE_DIR" ]]; then
  echo "WFLOW_SUITE_DIR must point to the completed W-Flow checkpoint suite." >&2
  exit 1
fi
if [[ -z "$BASELINE_SUITE_DIR" ]]; then
  echo "BASELINE_SUITE_DIR must point to the completed WGAN/diffusion implant suite." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$WALLCLOCK_SUITE_DIR"

IFS=',' read -r -a CHANNEL_ARRAY <<< "$CHANNELS"
CHANNEL_COUNT="${#CHANNEL_ARRAY[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT=$((NUM_SEEDS * CHANNEL_COUNT))

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] wflow_suite_dir: $WFLOW_SUITE_DIR"
echo "[slurm] wflow_suite_dir_map: ${WFLOW_SUITE_DIR_MAP:-none}"
echo "[slurm] baseline_suite_dir: $BASELINE_SUITE_DIR"
echo "[slurm] wallclock_suite_dir: $WALLCLOCK_SUITE_DIR"
echo "[slurm] train_seconds: $TRAIN_SECONDS"
echo "[slurm] seeds: $SEED_START..$((SEED_START + NUM_SEEDS - 1))"
echo "[slurm] channels: $CHANNELS"
echo "[slurm] variants: $VARIANTS"
echo "[slurm] log_every: $LOG_EVERY"
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
    "$PYTHON_BIN" -u scripts/run_equal_wallclock_symbolic_implant_seed_channel.py
    --device cuda
    --seed "$CURRENT_SEED"
    --channel "$CURRENT_CHANNEL"
    --variants "$VARIANTS"
    --wflow-suite-dir "$WFLOW_SUITE_DIR"
    --baseline-suite-dir "$BASELINE_SUITE_DIR"
    --suite-dir "$WALLCLOCK_SUITE_DIR"
    --train-seconds "$TRAIN_SECONDS"
    --batch-size "$AE_BATCH_SIZE"
    --learning-rate "$AE_LEARNING_RATE"
    --eval-size "$EVAL_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --diffusion-ddim-steps "$DIFFUSION_DDIM_STEPS"
    --log-every "$LOG_EVERY"
  )
  if [[ -n "$WFLOW_SUITE_DIR_MAP" ]]; then
    CMD+=(--wflow-suite-dir-map "$WFLOW_SUITE_DIR_MAP")
  fi
  if [[ "$SAVE_CHECKPOINTS" == "1" || "$SAVE_CHECKPOINTS" == "true" || "$SAVE_CHECKPOINTS" == "yes" ]]; then
    CMD+=(--save-checkpoints)
  fi

  "${CMD[@]}"
done
