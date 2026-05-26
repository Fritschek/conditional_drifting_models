#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHANNELS="${REPAIR_CHANNELS:-AWGN,Rayleigh,SSPA,OptFib}"
WALLTIME="${WALLTIME:-3-00:00:00}"
REPAIR_TASKS="${REPAIR_TASKS:-joint_sinkhorn:103 fiber_sinkhorn:103 fiber_sinkhorn:104 fiber_sinkhorn:105 fiber_sinkhorn:106}"
SLURM_SCRIPT="$PROJECT_ROOT/hpc/slurm_journal_wflow_repair_task.sh"
HPC_USE_CONDA="${HPC_USE_CONDA:-0}"

if [[ -z "${SUITE_DIR:-}" ]]; then
  SUITE_DIR="$(
    {
      find "$PROJECT_ROOT/results" -maxdepth 1 -type d \( -name 'journal_wflow_paper_hpc_*' -o -name 'journal_wflow_hpc_*' \) -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -1 | cut -d' ' -f2-
  )"
  if [[ -z "$SUITE_DIR" ]]; then
    echo "SUITE_DIR must be set to the W-Flow suite being repaired." >&2
    exit 1
  fi
  echo "[submit] auto-detected SUITE_DIR=$SUITE_DIR"
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SUITE_DIR"

export PROJECT_ROOT
export SUITE_DIR
export CHANNELS
export HPC_USE_CONDA

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] suite_dir: $SUITE_DIR"
echo "[submit] channels: $CHANNELS"
echo "[submit] walltime: $WALLTIME"
echo "[submit] repair_tasks: $REPAIR_TASKS"

for task in $REPAIR_TASKS; do
  variant="${task%%:*}"
  seed="${task##*:}"
  if [[ -z "$variant" || -z "$seed" || "$variant" == "$task" ]]; then
    echo "Invalid repair task '$task'; expected variant:seed." >&2
    exit 1
  fi

  job_id="$(
    sbatch --parsable \
      --chdir="$PROJECT_ROOT" \
      --time="$WALLTIME" \
      --job-name="wflow_fix_${variant}_${seed}" \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",SUITE_DIR="$SUITE_DIR",CHANNELS="$CHANNELS",REPAIR_VARIANT="$variant",REPAIR_SEED="$seed",HPC_USE_CONDA="$HPC_USE_CONDA" \
      "$SLURM_SCRIPT"
  )"
  echo "[submit] $variant seed $seed -> $job_id"
done
