#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "$PROJECT_ROOT/hpc/load_env.sh"

echo "[check] project_root: $PROJECT_ROOT"
echo "[check] python_bin: $PYTHON_BIN"
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

echo "[check] environment looks usable"
