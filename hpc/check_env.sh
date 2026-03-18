#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_SMOKE_CPUS="${SLURM_SMOKE_CPUS:-4}"
SLURM_SMOKE_MEM="${SLURM_SMOKE_MEM:-16G}"
SLURM_SMOKE_TIME="${SLURM_SMOKE_TIME:-00:15:00}"
SLURM_SMOKE_GPU_COUNT="${SLURM_SMOKE_GPU_COUNT:-1}"

RUN_LOCAL_ONLY=0
RUN_SLURM_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-only)
      RUN_LOCAL_ONLY=1
      shift
      ;;
    --slurm-smoke)
      RUN_SLURM_SMOKE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$RUN_SLURM_SMOKE" -eq 1 && "$RUN_LOCAL_ONLY" -eq 0 && -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[check] launching SLURM smoke allocation via srun"
  echo "[check] cpus=$SLURM_SMOKE_CPUS mem=$SLURM_SMOKE_MEM time=$SLURM_SMOKE_TIME gpus=$SLURM_SMOKE_GPU_COUNT"
  cd "$PROJECT_ROOT"
  exec srun \
    --gres="gpu:${SLURM_SMOKE_GPU_COUNT}" \
    --cpus-per-task="$SLURM_SMOKE_CPUS" \
    --mem="$SLURM_SMOKE_MEM" \
    --time="$SLURM_SMOKE_TIME" \
    bash hpc/check_env.sh --local-only
fi

source "$PROJECT_ROOT/hpc/load_env.sh"

echo "[check] project_root: $PROJECT_ROOT"
echo "[check] python_bin: $PYTHON_BIN"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[check] slurm_job_id: $SLURM_JOB_ID"
  echo "[check] running inside SLURM allocation"
else
  echo "[check] running outside SLURM allocation"
fi
echo

if command -v module >/dev/null 2>&1; then
  echo "[check] module command is available"
  echo "[check] loaded modules:"
  module list 2>&1 || true
else
  echo "[check] module command not found"
fi
echo

echo "[check] python path"
which "$PYTHON_BIN"
"$PYTHON_BIN" -V
echo

echo "[check] torch / cuda"
"$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"torch_import_error: {exc}")
    sys.exit(1)

print(f"torch_version: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    try:
        print(f"device_0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"device_name_error: {exc}")
PY
echo

echo "[check] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi
echo

echo "[check] repo import"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" - <<'PY'
import sys

try:
    import conditional_drifting
except Exception as exc:
    print(f"repo_import_error: {exc}")
    sys.exit(1)

print("repo_import_ok: conditional_drifting")
PY
echo

echo "[check] small benchmark smoke test"
"$PYTHON_BIN" -u scripts/run_full_budget_seed.py \
  --device cuda \
  --seed 7 \
  --paper-channels AWGN \
  --paper-eval-size 1000 \
  --optfib-eval-size 1000 \
  --suite-dir /tmp/conditional_drifting_hpc_smoke
echo

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[check] local shell check finished"
  echo "[check] for a real SLURM smoke test, run:"
  echo "bash hpc/check_env.sh --slurm-smoke"
else
  echo "[check] SLURM smoke test finished"
fi
