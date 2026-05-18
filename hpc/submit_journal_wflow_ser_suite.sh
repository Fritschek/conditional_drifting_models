#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-100}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,OptFib}"
VARIANTS="${VARIANTS:-analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SER_SUITE_DIR="${SER_SUITE_DIR:-$PROJECT_ROOT/results/journal_wflow_ser_${SUITE_TAG}}"
AE_DATASET_SIZE="${AE_DATASET_SIZE:-1000000}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-500}"
AE_EPOCHS="${AE_EPOCHS:-10}"
AE_LEARNING_RATE="${AE_LEARNING_RATE:-0.001}"
EVAL_SIZE="${EVAL_SIZE:-100000}"
EVAL_EVERY="${EVAL_EVERY:-1}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-32}"
MAX_PARALLEL="${MAX_PARALLEL:-}"

if [[ -z "${WFLOW_SUITE_DIR:-}" ]]; then
  WFLOW_SUITE_DIR="$(
    {
      find "$PROJECT_ROOT/results" -maxdepth 1 -type d \( -name 'journal_wflow_paper_hpc_*' -o -name 'journal_wflow_hpc_*' \) -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -1 | cut -d' ' -f2-
  )"
  if [[ -z "$WFLOW_SUITE_DIR" ]]; then
    echo "WFLOW_SUITE_DIR must point to the completed journal W-Flow checkpoint suite." >&2
    exit 1
  fi
  echo "[submit] auto-detected WFLOW_SUITE_DIR=$WFLOW_SUITE_DIR"
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SER_SUITE_DIR"

IFS=',' read -r -a CHANNEL_ARRAY <<< "$CHANNELS"
CHANNEL_COUNT="${#CHANNEL_ARRAY[@]}"
if [[ "$CHANNEL_COUNT" -lt 1 ]]; then
  echo "CHANNELS must contain at least one channel" >&2
  exit 1
fi

TASK_COUNT=$((NUM_SEEDS * CHANNEL_COUNT))
if (( TASK_COUNT < 1 )); then
  echo "No BER/SER tasks to submit" >&2
  exit 1
fi
if (( MAX_ARRAY_TASKS < 1 )); then
  echo "MAX_ARRAY_TASKS must be at least 1" >&2
  exit 1
fi

ARRAY_TASK_COUNT="$TASK_COUNT"
if (( ARRAY_TASK_COUNT > MAX_ARRAY_TASKS )); then
  ARRAY_TASK_COUNT="$MAX_ARRAY_TASKS"
fi

ARRAY_END=$((ARRAY_TASK_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}"
if [[ -n "$MAX_PARALLEL" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"
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
export ARRAY_TASK_COUNT

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] wflow_suite_dir: $WFLOW_SUITE_DIR"
echo "[submit] ser_suite_dir: $SER_SUITE_DIR"
echo "[submit] seeds: $SEED_START..$((SEED_START + NUM_SEEDS - 1))"
echo "[submit] channels: $CHANNELS"
echo "[submit] variants: $VARIANTS"
echo "[submit] ae_dataset_size: $AE_DATASET_SIZE"
echo "[submit] ae_batch_size: $AE_BATCH_SIZE"
echo "[submit] ae_epochs: $AE_EPOCHS"
echo "[submit] eval_size: $EVAL_SIZE"
echo "[submit] logical_seed_channel_tasks: $TASK_COUNT"
echo "[submit] max_array_tasks: $MAX_ARRAY_TASKS"
echo "[submit] submitted_array_tasks: $ARRAY_TASK_COUNT"
echo "[submit] array: $ARRAY_SPEC"

ARRAY_JOB_ID=$(sbatch --parsable --array="$ARRAY_SPEC" hpc/slurm_journal_wflow_ser_array.sh)
echo "[submit] array_job_id: $ARRAY_JOB_ID"
echo "[submit] aggregate after completion with:"
echo "export PROJECT_ROOT=\"$PROJECT_ROOT\""
echo "export SER_SUITE_DIR=\"$SER_SUITE_DIR\""
echo "export SEED_START=$SEED_START"
echo "export NUM_SEEDS=$NUM_SEEDS"
echo "export CHANNELS=\"$CHANNELS\""
echo "export VARIANTS=\"$VARIANTS\""
echo "sbatch --dependency=afterok:${ARRAY_JOB_ID} hpc/slurm_journal_wflow_ser_aggregate.sh"
