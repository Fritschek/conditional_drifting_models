#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-wgan,diffusion}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
BASELINE_SUITE_DIR="${BASELINE_SUITE_DIR:-$PROJECT_ROOT/results/journal_baseline_implants_${SUITE_TAG}}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-32}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
SLURM_TIME="${SLURM_TIME:-3-12:00:00}"
SLURM_SCRIPT="$PROJECT_ROOT/hpc/slurm_journal_baseline_implant_array.sh"
AGGREGATE_SCRIPT="$PROJECT_ROOT/hpc/slurm_journal_baseline_implant_aggregate.sh"
HPC_USE_CONDA="${HPC_USE_CONDA:-0}"
AUTO_AGGREGATE="${AUTO_AGGREGATE:-1}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$BASELINE_SUITE_DIR"

IFS=',' read -r -a CHANNEL_ARRAY <<< "$CHANNELS"
CHANNEL_COUNT="${#CHANNEL_ARRAY[@]}"
if [[ "$CHANNEL_COUNT" -lt 1 ]]; then
  echo "CHANNELS must contain at least one channel" >&2
  exit 1
fi

TASK_COUNT=$((NUM_SEEDS * CHANNEL_COUNT))
if (( TASK_COUNT < 1 )); then
  echo "No baseline implant tasks to submit" >&2
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
export BASELINE_SUITE_DIR
export ARRAY_TASK_COUNT
export HPC_USE_CONDA
export AUTO_AGGREGATE

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] baseline_suite_dir: $BASELINE_SUITE_DIR"
echo "[submit] seeds: $SEED_START..$((SEED_START + NUM_SEEDS - 1))"
echo "[submit] channels: $CHANNELS"
echo "[submit] variants: $VARIANTS"
echo "[submit] logical_seed_channel_tasks: $TASK_COUNT"
echo "[submit] max_array_tasks: $MAX_ARRAY_TASKS"
echo "[submit] submitted_array_tasks: $ARRAY_TASK_COUNT"
echo "[submit] array: $ARRAY_SPEC"
echo "[submit] slurm_time: $SLURM_TIME"

ARRAY_JOB_ID=$(sbatch --parsable --chdir="$PROJECT_ROOT" --time="$SLURM_TIME" --array="$ARRAY_SPEC" "$SLURM_SCRIPT")
echo "[submit] array_job_id: $ARRAY_JOB_ID"

if [[ "$AUTO_AGGREGATE" == "1" || "$AUTO_AGGREGATE" == "true" || "$AUTO_AGGREGATE" == "yes" ]]; then
  AGGREGATE_JOB_ID=$(sbatch --parsable --chdir="$PROJECT_ROOT" --dependency=afterok:${ARRAY_JOB_ID} "$AGGREGATE_SCRIPT")
  echo "[submit] aggregate_job_id: $AGGREGATE_JOB_ID"
else
  echo "[submit] aggregate after completion with:"
  echo "export PROJECT_ROOT=\"$PROJECT_ROOT\""
  echo "export BASELINE_SUITE_DIR=\"$BASELINE_SUITE_DIR\""
  echo "export SEED_START=$SEED_START"
  echo "export NUM_SEEDS=$NUM_SEEDS"
  echo "export CHANNELS=\"$CHANNELS\""
  echo "export VARIANTS=\"$VARIANTS\""
  echo "sbatch --chdir=\"$PROJECT_ROOT\" --dependency=afterok:${ARRAY_JOB_ID} \"$AGGREGATE_SCRIPT\""
fi
