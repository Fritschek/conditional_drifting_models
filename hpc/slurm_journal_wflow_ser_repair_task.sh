#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=cond_drift_ser_fix
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VARIANTS="${SER_REPAIR_VARIANTS:-${VARIANTS:-analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}}"
AE_DATASET_SIZE="${AE_DATASET_SIZE:-1000000}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-500}"
AE_EPOCHS="${AE_EPOCHS:-10}"
AE_LEARNING_RATE="${AE_LEARNING_RATE:-0.001}"
EVAL_SIZE="${EVAL_SIZE:-100000}"
EVAL_EVERY="${EVAL_EVERY:-1}"

if [[ -z "${WFLOW_SUITE_DIR:-}" ]]; then
  echo "WFLOW_SUITE_DIR must point to the completed journal W-Flow checkpoint suite." >&2
  exit 1
fi
if [[ -z "${SER_SUITE_DIR:-}" ]]; then
  echo "SER_SUITE_DIR must be set to the BER/SER suite being repaired." >&2
  exit 1
fi
if [[ -z "${REPAIR_CHANNEL:-}" || -z "${REPAIR_SEED:-}" ]]; then
  echo "REPAIR_CHANNEL and REPAIR_SEED must be set." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SER_SUITE_DIR"

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] wflow_suite_dir: $WFLOW_SUITE_DIR"
echo "[slurm] ser_suite_dir: $SER_SUITE_DIR"
echo "[slurm] channel: $REPAIR_CHANNEL"
echo "[slurm] seed: $REPAIR_SEED"
echo "[slurm] variants: $VARIANTS"
echo "[slurm] job_id: ${SLURM_JOB_ID:-n/a}"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

echo "[slurm] GPU information"
nvidia-smi || true

"$PYTHON_BIN" -u scripts/run_journal_wflow_ser_seed_channel.py \
  --device cuda \
  --seed "$REPAIR_SEED" \
  --channel "$REPAIR_CHANNEL" \
  --variants "$VARIANTS" \
  --wflow-suite-dir "$WFLOW_SUITE_DIR" \
  --suite-dir "$SER_SUITE_DIR" \
  --ae-dataset-size "$AE_DATASET_SIZE" \
  --ae-batch-size "$AE_BATCH_SIZE" \
  --ae-epochs "$AE_EPOCHS" \
  --ae-learning-rate "$AE_LEARNING_RATE" \
  --eval-size "$EVAL_SIZE" \
  --eval-every "$EVAL_EVERY"
