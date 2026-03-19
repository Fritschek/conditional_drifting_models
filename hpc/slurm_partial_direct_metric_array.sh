#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --job-name=cond_drift_partial
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-10}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_ROOT/results/partial_direct_metric_hpc_${SUITE_TAG}}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,OptFib}"
METHODS="${METHODS:-drifting_residual,wgan,optfib_diffusion}"
PAPER_EVAL_SIZE="${PAPER_EVAL_SIZE:-1000000}"
OPTFIB_EVAL_SIZE="${OPTFIB_EVAL_SIZE:-100000}"
OPTFIB_DATASET_SIZE="${OPTFIB_DATASET_SIZE:-120000}"
OPTFIB_EPOCHS="${OPTFIB_EPOCHS:-60}"
OPTFIB_BATCH_SIZE="${OPTFIB_BATCH_SIZE:-512}"
OPTFIB_NUM_STEPS="${OPTFIB_NUM_STEPS:-100}"

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

"$PYTHON_BIN" -u scripts/run_partial_direct_metric_seed.py \
  --device cuda \
  --seed "$CURRENT_SEED" \
  --channels "$CHANNELS" \
  --methods "$METHODS" \
  --paper-eval-size "$PAPER_EVAL_SIZE" \
  --optfib-eval-size "$OPTFIB_EVAL_SIZE" \
  --optfib-dataset-size "$OPTFIB_DATASET_SIZE" \
  --optfib-epochs "$OPTFIB_EPOCHS" \
  --optfib-batch-size "$OPTFIB_BATCH_SIZE" \
  --optfib-num-steps "$OPTFIB_NUM_STEPS" \
  --suite-dir "$SUITE_DIR"
