#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=cond_drift_base_swd_agg
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DIRECT_SWD_SUITE_DIR="${DIRECT_SWD_SUITE_DIR:?DIRECT_SWD_SUITE_DIR must point to the direct-SWD output directory.}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-TDL}"
VARIANTS="${VARIANTS:-wgan,ddpm,ddim100}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

mkdir -p "$PROJECT_ROOT/logs"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -u scripts/aggregate_journal_baseline_direct_swd.py
  --suite-dir "$DIRECT_SWD_SUITE_DIR"
  --channels "$CHANNELS"
  --variants "$VARIANTS"
  --seed-start "$SEED_START"
  --num-seeds "$NUM_SEEDS"
)

if [[ "$ALLOW_MISSING" == "1" || "$ALLOW_MISSING" == "true" || "$ALLOW_MISSING" == "yes" ]]; then
  CMD+=(--allow-missing)
fi

"${CMD[@]}"
