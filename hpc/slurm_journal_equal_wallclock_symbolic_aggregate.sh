#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=cond_drift_wall_agg
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-analytic,fiber_sinkhorn,wgan,diffusion_ddim100}"

if [[ -z "${WALLCLOCK_SUITE_DIR:-}" ]]; then
  echo "WALLCLOCK_SUITE_DIR must be set." >&2
  exit 1
fi

source "$PROJECT_ROOT/hpc/load_env.sh"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -u scripts/aggregate_equal_wallclock_symbolic_suite.py \
  --suite-dir "$WALLCLOCK_SUITE_DIR" \
  --seed-start "$SEED_START" \
  --num-seeds "$NUM_SEEDS" \
  --channels "$CHANNELS" \
  --variants "$VARIANTS"
