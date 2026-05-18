#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cond_drift_ser_agg
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-100}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,OptFib}"
VARIANTS="${VARIANTS:-analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
PAIRED_BASELINE="${PAIRED_BASELINE:-analytic}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

if [[ -z "${SER_SUITE_DIR:-}" ]]; then
  echo "SER_SUITE_DIR must be set to the shared BER/SER output directory." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -u scripts/aggregate_journal_wflow_ser_suite.py
  --suite-dir "$SER_SUITE_DIR"
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
