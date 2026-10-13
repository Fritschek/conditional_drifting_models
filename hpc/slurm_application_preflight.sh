#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=application_check
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HPC_USE_CONDA="${HPC_USE_CONDA:-0}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this preflight with sbatch inside the repository; it needs a GPU allocation." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/hpc/load_env.sh"

for path in results/official_fast_sources_20261012/guided_diffusion \
            results/official_fast_sources_20261012/flow_matching \
            results/official_fast_sources_20261012/consistency_models \
            results/official_sources_20261010/W-Flow; do
  if [[ ! -d "$path" ]]; then
    echo "Missing pinned source checkout: $path" >&2
    echo "Transfer the source directories, or explicitly run scripts/fetch_fast_baseline_sources.py first." >&2
    echo "This job performs no downloads, installations or upgrades." >&2
    exit 1
  fi
done

SUITE_DIR="$PROJECT_ROOT/results/application_preflight_${SLURM_JOB_ID}"
echo "[preflight] root: $PROJECT_ROOT"
echo "[preflight] suite_dir: $SUITE_DIR"
"$PYTHON_BIN" -c 'import torch; print("torch:", torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

export FAST_TEST_DEVICE=cuda
"$PYTHON_BIN" -m unittest discover -s tests -p test_application_local.py

"$PYTHON_BIN" -u scripts/run_common_coverage_application.py \
  --profile smoke --seed 9399 --out-dir "$SUITE_DIR/sspa"
"$PYTHON_BIN" -u scripts/run_common_coverage_application.py \
  --profile smoke --seed 9399 --out-dir "$SUITE_DIR/sspa" --resume
"$PYTHON_BIN" scripts/report_application_local.py --kind sspa --suite-dir "$SUITE_DIR/sspa"

"$PYTHON_BIN" -u scripts/run_structured_ofdm_frequency_local.py \
  --profile smoke --seed 9499 --condition interference --learn-constellation \
  --families gaussian,mixture,sinkhorn,flow,consistency_distilled,wgan,diffusion \
  --out-dir "$SUITE_DIR/ofdm"
"$PYTHON_BIN" -u scripts/run_structured_ofdm_frequency_local.py \
  --profile smoke --seed 9499 --condition interference --learn-constellation \
  --families gaussian,mixture,sinkhorn,flow,consistency_distilled,wgan,diffusion \
  --out-dir "$SUITE_DIR/ofdm" --resume
"$PYTHON_BIN" scripts/report_application_local.py --kind ofdm --suite-dir "$SUITE_DIR/ofdm"

echo "[preflight] complete; this is an environment/execution check, not a scientific campaign."
