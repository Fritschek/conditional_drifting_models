#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cond_drift_partial_agg
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-10}"

if [[ -z "${SUITE_DIR:-}" ]]; then
  echo "SUITE_DIR must be set to the shared output directory used by the array jobs." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -u scripts/aggregate_partial_direct_metric_suite.py \
  --suite-dir "$SUITE_DIR" \
  --seed-start "$SEED_START" \
  --num-seeds "$NUM_SEEDS"
