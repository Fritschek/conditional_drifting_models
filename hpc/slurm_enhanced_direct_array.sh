#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=cond_drift_enh
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-10}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_ROOT/results/enhanced_direct_hpc_${SUITE_TAG}}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA}"
DATASET_SIZE="${DATASET_SIZE:--1}"
EVAL_SIZE="${EVAL_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:--1}"
DRIFTING_EPOCHS="${DRIFTING_EPOCHS:--1}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:--1}"
CONDITIONING_MODE="${CONDITIONING_MODE:-joint}"
CONDITION_KERNEL_SCALE="${CONDITION_KERNEL_SCALE:-0.5}"
TARGET_KERNEL_SCALE="${TARGET_KERNEL_SCALE:-1.0}"
TARGET_KERNEL_MODE="${TARGET_KERNEL_MODE:-raw}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SUITE_DIR"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  CURRENT_SEED=$((SEED_START + SLURM_ARRAY_TASK_ID))
else
  CURRENT_SEED="${CURRENT_SEED:-$SEED_START}"
fi

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] suite_dir: $SUITE_DIR"
echo "[slurm] seed: $CURRENT_SEED"
echo "[slurm] job_id: ${SLURM_JOB_ID:-n/a}"
echo "[slurm] array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

echo "[slurm] GPU information"
nvidia-smi || true

"$PYTHON_BIN" -u scripts/run_enhanced_direct_seed.py \
  --device cuda \
  --seed "$CURRENT_SEED" \
  --channels "$CHANNELS" \
  --dataset-size "$DATASET_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --drifting-epochs "$DRIFTING_EPOCHS" \
  --swd-projections "$SWD_PROJECTIONS" \
  --conditioning-mode "$CONDITIONING_MODE" \
  --condition-kernel-scale "$CONDITION_KERNEL_SCALE" \
  --target-kernel-scale "$TARGET_KERNEL_SCALE" \
  --target-kernel-mode "$TARGET_KERNEL_MODE" \
  --suite-dir "$SUITE_DIR"
