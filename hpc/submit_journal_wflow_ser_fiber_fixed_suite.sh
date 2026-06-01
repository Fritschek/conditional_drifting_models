#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-100}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-analytic,fiber_sinkhorn}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SER_SUITE_DIR="${SER_SUITE_DIR:-$PROJECT_ROOT/results/journal_wflow_ser_fiber_fixed_${SUITE_TAG}}"
AE_DATASET_SIZE="${AE_DATASET_SIZE:-1000000}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-500}"
AE_EPOCHS="${AE_EPOCHS:-10}"
AE_LEARNING_RATE="${AE_LEARNING_RATE:-0.001}"
EVAL_SIZE="${EVAL_SIZE:-100000}"
EVAL_EVERY="${EVAL_EVERY:-1}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-32}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
HPC_USE_CONDA="${HPC_USE_CONDA:-0}"

if [[ -z "${WFLOW_SUITE_DIR:-}" ]]; then
  echo "WFLOW_SUITE_DIR must point to the corrected fiber Sinkhorn W-Flow suite." >&2
  echo "Example: export WFLOW_SUITE_DIR=\"$PROJECT_ROOT/results/journal_wflow_fiber_fixed_YYYYMMDD_HHMMSS\"" >&2
  exit 1
fi

export PROJECT_ROOT
export SEED_START
export NUM_SEEDS
export CHANNELS
export VARIANTS
export WFLOW_SUITE_DIR
export SER_SUITE_DIR
export AE_DATASET_SIZE
export AE_BATCH_SIZE
export AE_EPOCHS
export AE_LEARNING_RATE
export EVAL_SIZE
export EVAL_EVERY
export MAX_ARRAY_TASKS
export MAX_PARALLEL
export HPC_USE_CONDA

"$PROJECT_ROOT/hpc/submit_journal_wflow_ser_suite.sh"
