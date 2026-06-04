#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
CHANNELS="${CHANNELS:-AWGN,Rayleigh,SSPA,TDL}"
VARIANTS="${VARIANTS:-analytic,fiber_sinkhorn,wgan,diffusion_ddim100}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
WALLCLOCK_SUITE_DIR="${WALLCLOCK_SUITE_DIR:-$PROJECT_ROOT/results/journal_equal_wallclock_symbolic_${SUITE_TAG}}"
TRAIN_SECONDS="${TRAIN_SECONDS:-1800}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-32}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
SLURM_TIME="${SLURM_TIME:-0-06:00:00}"
HPC_USE_CONDA="${HPC_USE_CONDA:-0}"
AUTO_AGGREGATE="${AUTO_AGGREGATE:-1}"
WFLOW_SUITE_DIR="${WFLOW_SUITE_DIR:-}"
WFLOW_SUITE_DIR_MAP="${WFLOW_SUITE_DIR_MAP:-}"
BASELINE_SUITE_DIR="${BASELINE_SUITE_DIR:-}"
DIFFUSION_DDIM_STEPS="${DIFFUSION_DDIM_STEPS:-100}"
LOG_EVERY="${LOG_EVERY:-500}"
SLURM_SCRIPT="$PROJECT_ROOT/hpc/slurm_journal_equal_wallclock_symbolic_array.sh"
AGGREGATE_SCRIPT="$PROJECT_ROOT/hpc/slurm_journal_equal_wallclock_symbolic_aggregate.sh"

if [[ -z "$WFLOW_SUITE_DIR" ]]; then
  WFLOW_SUITE_DIR="$(
    {
      find "$PROJECT_ROOT/results" -maxdepth 1 -type d \( -name 'journal_wflow_fiber_fixed_paper_hpc_*' -o -name 'journal_wflow_paper_hpc_*' \) -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -1 | cut -d' ' -f2-
  )"
  if [[ -z "$WFLOW_SUITE_DIR" ]]; then
    echo "WFLOW_SUITE_DIR must point to the completed W-Flow checkpoint suite." >&2
    exit 1
  fi
  echo "[submit] auto-detected WFLOW_SUITE_DIR=$WFLOW_SUITE_DIR"
fi

if [[ -z "$BASELINE_SUITE_DIR" ]]; then
  BASELINE_SUITE_DIR="$(
    {
      find "$PROJECT_ROOT/results" -maxdepth 1 -type d -name 'journal_baseline_implants_*' -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -1 | cut -d' ' -f2-
  )"
  if [[ -z "$BASELINE_SUITE_DIR" ]]; then
    echo "BASELINE_SUITE_DIR must point to the completed WGAN/diffusion implant suite." >&2
    exit 1
  fi
  echo "[submit] auto-detected BASELINE_SUITE_DIR=$BASELINE_SUITE_DIR"
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$WALLCLOCK_SUITE_DIR"

IFS=',' read -r -a CHANNEL_ARRAY <<< "$CHANNELS"
CHANNEL_COUNT="${#CHANNEL_ARRAY[@]}"
TASK_COUNT=$((NUM_SEEDS * CHANNEL_COUNT))
if (( TASK_COUNT < 1 )); then
  echo "No equal-wall-clock tasks to submit" >&2
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
ARRAY_SPEC="0-$((ARRAY_TASK_COUNT - 1))"
if [[ -n "$MAX_PARALLEL" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"
fi

export PROJECT_ROOT
export SEED_START
export NUM_SEEDS
export CHANNELS
export VARIANTS
export WFLOW_SUITE_DIR
export WFLOW_SUITE_DIR_MAP
export BASELINE_SUITE_DIR
export WALLCLOCK_SUITE_DIR
export TRAIN_SECONDS
export ARRAY_TASK_COUNT
export HPC_USE_CONDA
export DIFFUSION_DDIM_STEPS
export LOG_EVERY

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] wallclock_suite_dir: $WALLCLOCK_SUITE_DIR"
echo "[submit] wflow_suite_dir: $WFLOW_SUITE_DIR"
echo "[submit] wflow_suite_dir_map: ${WFLOW_SUITE_DIR_MAP:-none}"
echo "[submit] baseline_suite_dir: $BASELINE_SUITE_DIR"
echo "[submit] seeds: $SEED_START..$((SEED_START + NUM_SEEDS - 1))"
echo "[submit] channels: $CHANNELS"
echo "[submit] variants: $VARIANTS"
echo "[submit] train_seconds_per_variant: $TRAIN_SECONDS"
echo "[submit] log_every: $LOG_EVERY"
echo "[submit] logical_seed_channel_tasks: $TASK_COUNT"
echo "[submit] max_array_tasks: $MAX_ARRAY_TASKS"
echo "[submit] submitted_array_tasks: $ARRAY_TASK_COUNT"
echo "[submit] slurm_time: $SLURM_TIME"
echo "[submit] array: $ARRAY_SPEC"

ARRAY_JOB_ID=$(sbatch --parsable --chdir="$PROJECT_ROOT" --time="$SLURM_TIME" --array="$ARRAY_SPEC" "$SLURM_SCRIPT")
echo "[submit] array_job_id: $ARRAY_JOB_ID"

if [[ "$AUTO_AGGREGATE" == "1" || "$AUTO_AGGREGATE" == "true" || "$AUTO_AGGREGATE" == "yes" ]]; then
  AGGREGATE_JOB_ID=$(sbatch --parsable --chdir="$PROJECT_ROOT" --dependency=afterok:${ARRAY_JOB_ID} "$AGGREGATE_SCRIPT")
  echo "[submit] aggregate_job_id: $AGGREGATE_JOB_ID"
else
  echo "[submit] aggregate after completion with:"
  echo "export PROJECT_ROOT=\"$PROJECT_ROOT\""
  echo "export WALLCLOCK_SUITE_DIR=\"$WALLCLOCK_SUITE_DIR\""
  echo "export SEED_START=$SEED_START"
  echo "export NUM_SEEDS=$NUM_SEEDS"
  echo "export CHANNELS=\"$CHANNELS\""
  echo "export VARIANTS=\"$VARIANTS\""
  echo "sbatch --chdir=\"$PROJECT_ROOT\" --dependency=afterok:${ARRAY_JOB_ID} \"$AGGREGATE_SCRIPT\""
fi
