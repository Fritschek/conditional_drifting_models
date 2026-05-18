#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cond_drift_wflow_agg
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-100}"
VARIANTS="${VARIANTS:-kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,OptFib}"
PAIRED_BASELINE="${PAIRED_BASELINE:-kernel_joint}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

if [[ -z "${SUITE_DIR:-}" ]]; then
  echo "SUITE_DIR must be set to the shared output directory used by the array jobs." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -u scripts/aggregate_journal_wflow_suite.py
  --suite-dir "$SUITE_DIR"
  --variants "$VARIANTS"
  --channels "$CHANNELS"
  --seed-start "$SEED_START"
  --num-seeds "$NUM_SEEDS"
  --paired-baseline "$PAIRED_BASELINE"
)

if [[ "$ALLOW_MISSING" == "1" || "$ALLOW_MISSING" == "true" || "$ALLOW_MISSING" == "yes" ]]; then
  CMD+=(--allow-missing)
fi

"${CMD[@]}"
