#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cond_drift_base_agg
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-wgan,diffusion}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

if [[ -z "${BASELINE_SUITE_DIR:-}" ]]; then
  echo "BASELINE_SUITE_DIR must be set to the shared baseline implant output directory." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -u scripts/aggregate_journal_baseline_implant_suite.py
  --suite-dir "$BASELINE_SUITE_DIR"
  --variants "$VARIANTS"
  --channels "$CHANNELS"
  --seed-start "$SEED_START"
  --num-seeds "$NUM_SEEDS"
)

if [[ "$ALLOW_MISSING" == "1" || "$ALLOW_MISSING" == "true" || "$ALLOW_MISSING" == "yes" ]]; then
  CMD+=(--allow-missing)
fi

"${CMD[@]}"
