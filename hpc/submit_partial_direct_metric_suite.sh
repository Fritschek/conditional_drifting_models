#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-10}"
SUITE_TAG="${SUITE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_ROOT/results/partial_direct_metric_hpc_${SUITE_TAG}}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$SUITE_DIR"

ARRAY_END=$((NUM_SEEDS - 1))

export PROJECT_ROOT
export SEED_START
export NUM_SEEDS
export SUITE_DIR

echo "[submit] project_root: $PROJECT_ROOT"
echo "[submit] suite_dir: $SUITE_DIR"
echo "[submit] seeds: $SEED_START..$((SEED_START + NUM_SEEDS - 1))"

ARRAY_JOB_ID=$(sbatch --parsable --array=0-"$ARRAY_END" hpc/slurm_partial_direct_metric_array.sh)
echo "[submit] array_job_id: $ARRAY_JOB_ID"
echo "[submit] aggregate after completion with:"
echo "export PROJECT_ROOT=\"$PROJECT_ROOT\""
echo "export SUITE_DIR=\"$SUITE_DIR\""
echo "export SEED_START=$SEED_START"
echo "export NUM_SEEDS=$NUM_SEEDS"
echo "sbatch --dependency=afterok:${ARRAY_JOB_ID} hpc/slurm_partial_direct_metric_aggregate.sh"
