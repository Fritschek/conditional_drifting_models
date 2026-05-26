#!/bin/bash

set -euo pipefail

# Shared HPC environment bootstrap.
# Adjust the module lines below to match the target cluster.

# Required for deterministic CuBLAS operations when PyTorch deterministic
# algorithms are enabled on CUDA.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if command -v module >/dev/null 2>&1; then
  module --force purge || true
  module load "${HPC_MODULE_RELEASE:-release/24.04}" || true
  module load "${HPC_MODULE_TOOLCHAIN:-GCC/12.3.0}" || true
  module load "${HPC_MODULE_MPI:-OpenMPI/4.1.5}" || true
  module load "${HPC_MODULE_PYTORCH:-PyTorch/2.1.2-CUDA-12.1.1}" || true
fi

HPC_USE_CONDA="${HPC_USE_CONDA:-0}"
if [[ "$HPC_USE_CONDA" == "1" || "$HPC_USE_CONDA" == "true" || "$HPC_USE_CONDA" == "yes" ]]; then
  if [[ -z "${CONDA_ENV:-}" ]]; then
    echo "[load_env] HPC_USE_CONDA is enabled but CONDA_ENV is not set" >&2
    exit 1
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  else
    echo "[load_env] CONDA_ENV is set but conda is not available" >&2
    exit 1
  fi
elif [[ -n "${CONDA_ENV:-}" ]]; then
  echo "[load_env] ignoring inherited CONDA_ENV=$CONDA_ENV because HPC_USE_CONDA is not enabled" >&2
fi
