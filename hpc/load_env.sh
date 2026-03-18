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

if [[ -n "${CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  else
    echo "[load_env] CONDA_ENV is set but conda is not available" >&2
    exit 1
  fi
fi
