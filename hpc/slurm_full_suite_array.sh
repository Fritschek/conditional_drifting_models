#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --job-name=cond_drift_suite
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# User-adjustable settings.
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_ROOT/results/hpc_full_budget_${SUITE_TAG}}"
PAPER_CHANNELS="${PAPER_CHANNELS:-AWGN,Rayleigh,SSPA}"
PAPER_EVAL_SIZE="${PAPER_EVAL_SIZE:-1000000}"
OPTFIB_EVAL_SIZE="${OPTFIB_EVAL_SIZE:-100000}"

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

"$PYTHON_BIN" -u scripts/run_full_budget_seed.py \
  --device cuda \
  --seed "$CURRENT_SEED" \
  --paper-channels "$PAPER_CHANNELS" \
  --paper-eval-size "$PAPER_EVAL_SIZE" \
  --optfib-eval-size "$OPTFIB_EVAL_SIZE" \
  --suite-dir "$SUITE_DIR"
