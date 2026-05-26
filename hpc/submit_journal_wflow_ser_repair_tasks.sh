#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VARIANTS="${VARIANTS:-analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn}"
WALLTIME="${WALLTIME:-1-12:00:00}"
REPAIR_TASKS="${REPAIR_TASKS:-Rayleigh:103 Rayleigh:104 Rayleigh:105 Rayleigh:106 SSPA:103 SSPA:104 SSPA:105 SSPA:106 OptFib:103 OptFib:104 OptFib:105 OptFib:106}"

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

if [[ -z "${SER_SUITE_DIR:-}" ]]; then
  SER_SUITE_DIR="$(
    {
      find "$PROJECT_ROOT/results" -maxdepth 1 -type d -name 'journal_wflow_ser_*' -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -1 | cut -d' ' -f2-
  )"
  if [[ -z "$SER_SUITE_DIR" ]]; then
    echo "SER_SUITE_DIR must be set to the BER/SER suite being repaired." >&2
    exit 1
  fi
  echo "[submit] auto-detected SER_SUITE_DIR=$SER_SUITE_DIR"
fi

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SER_SUITE_DIR"

export PROJECT_ROOT
export WFLOW_SUITE_DIR
export SER_SUITE_DIR
export VARIANTS

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] wflow_suite_dir: $WFLOW_SUITE_DIR"
echo "[submit] ser_suite_dir: $SER_SUITE_DIR"
echo "[submit] variants: $VARIANTS"
echo "[submit] walltime: $WALLTIME"
echo "[submit] repair_tasks: $REPAIR_TASKS"

for task in $REPAIR_TASKS; do
  channel="${task%%:*}"
  seed="${task##*:}"
  if [[ -z "$channel" || -z "$seed" || "$channel" == "$task" ]]; then
    echo "Invalid repair task '$task'; expected channel:seed." >&2
    exit 1
  fi

  job_id="$(
    sbatch --parsable \
      --time="$WALLTIME" \
      --job-name="ser_fix_${channel}_${seed}" \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",WFLOW_SUITE_DIR="$WFLOW_SUITE_DIR",SER_SUITE_DIR="$SER_SUITE_DIR",VARIANTS="$VARIANTS",REPAIR_CHANNEL="$channel",REPAIR_SEED="$seed" \
      hpc/slurm_journal_wflow_ser_repair_task.sh
  )"
  echo "[submit] $channel seed $seed -> $job_id"
done
