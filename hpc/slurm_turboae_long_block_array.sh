#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cond_drift_turboae
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TURBO_ROOT="${TURBO_ROOT:-$PROJECT_ROOT/external/turbo_mingru_decoder}"
SEED_START="${SEED_START:-7}"
NUM_SEEDS="${NUM_SEEDS:-30}"
TURBOAE_SUITE_DIR="${TURBOAE_SUITE_DIR:-$PROJECT_ROOT/results/turboae_long_block_hpc_$(date -u +%Y%m%d_%H%M%S)}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

TURBOAE_CHANNEL="${TURBOAE_CHANNEL:-AWGN}"
TURBOAE_LENGTHS="${TURBOAE_LENGTHS:-64}"
TURBOAE_MODES="${TURBOAE_MODES:-analytic,checkpoint}"
TURBOAE_MODEL_TYPE="${TURBOAE_MODEL_TYPE:-cnn_turbo}"
TURBOAE_EPOCHS="${TURBOAE_EPOCHS:-300}"
TURBOAE_BATCH_SIZE="${TURBOAE_BATCH_SIZE:-500}"
TURBOAE_DEC_BS_FAC="${TURBOAE_DEC_BS_FAC:-1}"
TURBOAE_ENC_MICRO_BATCH_SIZE="${TURBOAE_ENC_MICRO_BATCH_SIZE:-0}"
TURBOAE_DEC_MICRO_BATCH_SIZE="${TURBOAE_DEC_MICRO_BATCH_SIZE:-0}"
TURBOAE_SAMPLE_SIZE="${TURBOAE_SAMPLE_SIZE:-50000}"
TURBOAE_EVAL_NUM_BLOCKS="${TURBOAE_EVAL_NUM_BLOCKS:-50000}"
TURBOAE_EVAL_BATCHES="${TURBOAE_EVAL_BATCHES:-20}"
TURBOAE_EVAL_EVERY="${TURBOAE_EVAL_EVERY:-10}"
TURBOAE_SAVE_EVERY="${TURBOAE_SAVE_EVERY:-25}"
TURBOAE_LEARNING_RATE="${TURBOAE_LEARNING_RATE:-0.0002}"
TURBOAE_WEIGHT_DECAY="${TURBOAE_WEIGHT_DECAY:-0.01}"
TURBOAE_GRAD_CLIP_NORM="${TURBOAE_GRAD_CLIP_NORM:-1.0}"
TURBOAE_EBNO_DB="${TURBOAE_EBNO_DB:-4.0}"
TURBOAE_RATE="${TURBOAE_RATE:-0.5}"
TURBOAE_DECODER_EBNO_OFFSET_LOW="${TURBOAE_DECODER_EBNO_OFFSET_LOW:--3.5}"
TURBOAE_DECODER_EBNO_OFFSET_HIGH="${TURBOAE_DECODER_EBNO_OFFSET_HIGH:-0.0}"
TURBOAE_ALLOW_TF32="${TURBOAE_ALLOW_TF32:-1}"
TURBOAE_AMP="${TURBOAE_AMP:-0}"
TURBOAE_AMP_DTYPE="${TURBOAE_AMP_DTYPE:-bfloat16}"
TURBOAE_COMPILE_MODELS="${TURBOAE_COMPILE_MODELS:-0}"
TURBOAE_ENABLE_GPU_WARMUP="${TURBOAE_ENABLE_GPU_WARMUP:-0}"
TURBOAE_LR_PLATEAU_PATIENCE_EVALS="${TURBOAE_LR_PLATEAU_PATIENCE_EVALS:-0}"
TURBOAE_LR_PLATEAU_FACTOR="${TURBOAE_LR_PLATEAU_FACTOR:-0.5}"
TURBOAE_LR_PLATEAU_MAX_REDUCTIONS="${TURBOAE_LR_PLATEAU_MAX_REDUCTIONS:-0}"
TURBOAE_LR_PLATEAU_MIN_LR="${TURBOAE_LR_PLATEAU_MIN_LR:-0.00001}"

TRAIN_CHANNEL_IMPLANT="${TRAIN_CHANNEL_IMPLANT:-1}"
CHANNEL_IMPLANT_CHECKPOINT="${CHANNEL_IMPLANT_CHECKPOINT:-}"
CHANNEL_IMPLANT_DATASET_SIZE="${CHANNEL_IMPLANT_DATASET_SIZE:-120000}"
CHANNEL_IMPLANT_EVAL_SIZE="${CHANNEL_IMPLANT_EVAL_SIZE:-100000}"
CHANNEL_IMPLANT_BATCH_SIZE="${CHANNEL_IMPLANT_BATCH_SIZE:-512}"
CHANNEL_IMPLANT_EPOCHS="${CHANNEL_IMPLANT_EPOCHS:-30}"
CHANNEL_IMPLANT_SWD_PROJECTIONS="${CHANNEL_IMPLANT_SWD_PROJECTIONS:-128}"
CHANNEL_IMPLANT_ANCHOR_COUNT="${CHANNEL_IMPLANT_ANCHOR_COUNT:-128}"
CHANNEL_IMPLANT_ANCHOR_SAMPLES="${CHANNEL_IMPLANT_ANCHOR_SAMPLES:-64}"
CHANNEL_IMPLANT_ANCHOR_SWD_PROJECTIONS="${CHANNEL_IMPLANT_ANCHOR_SWD_PROJECTIONS:-64}"
CHANNEL_IMPLANT_SINKHORN_MIN_EPSILON="${CHANNEL_IMPLANT_SINKHORN_MIN_EPSILON:-0.001}"
CHANNEL_IMPLANT_SINKHORN_ITERATIONS="${CHANNEL_IMPLANT_SINKHORN_ITERATIONS:-10}"
CHANNEL_IMPLANT_FIBER_GENERATED_SAMPLES="${CHANNEL_IMPLANT_FIBER_GENERATED_SAMPLES:-4}"
CHANNEL_IMPLANT_FIBER_POSITIVE_SAMPLES="${CHANNEL_IMPLANT_FIBER_POSITIVE_SAMPLES:-4}"
CHANNEL_IMPLANT_FIBER_REFERENCE_SAMPLES="${CHANNEL_IMPLANT_FIBER_REFERENCE_SAMPLES:-4}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$TURBOAE_SUITE_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "[slurm] host: $(hostname)"
echo "[slurm] project_root: $PROJECT_ROOT"
echo "[slurm] turbo_root: $TURBO_ROOT"
echo "[slurm] suite_dir: $TURBOAE_SUITE_DIR"
echo "[slurm] seed_start: $SEED_START"
echo "[slurm] num_seeds: $NUM_SEEDS"
echo "[slurm] array_task_count: $ARRAY_TASK_COUNT"
echo "[slurm] lengths: $TURBOAE_LENGTHS"
echo "[slurm] modes: $TURBOAE_MODES"
echo "[slurm] channel: $TURBOAE_CHANNEL"
echo "[slurm] job_id: ${SLURM_JOB_ID:-n/a}"
echo "[slurm] array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"

source "$PROJECT_ROOT/hpc/load_env.sh"

cd "$PROJECT_ROOT"

echo "[slurm] GPU information"
nvidia-smi || true

contains_mode() {
  local needle="$1"
  local haystack=",$TURBOAE_MODES,"
  [[ "$haystack" == *",$needle,"* ]]
}

is_enabled() {
  local value="$1"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" ]]
}

for ((SEED_OFFSET = TASK_ID; SEED_OFFSET < NUM_SEEDS; SEED_OFFSET += ARRAY_TASK_COUNT)); do
  CURRENT_SEED=$((SEED_START + SEED_OFFSET))
  SEED_DIR="$TURBOAE_SUITE_DIR/seed${CURRENT_SEED}"
  IMPLANT_DIR="$SEED_DIR/channel_implants/awgn2_fiber_sinkhorn"
  IMPLANT_CHECKPOINT="$CHANNEL_IMPLANT_CHECKPOINT"

  mkdir -p "$SEED_DIR"
  echo "[slurm] running seed=$CURRENT_SEED seed_offset=$SEED_OFFSET"

  if contains_mode "checkpoint"; then
    if [[ -z "$IMPLANT_CHECKPOINT" ]]; then
      IMPLANT_CHECKPOINT="$IMPLANT_DIR/checkpoints/enhanced_direct_awgn_seed${CURRENT_SEED}.pt"
      if is_enabled "$TRAIN_CHANNEL_IMPLANT" || [[ ! -f "$IMPLANT_CHECKPOINT" ]]; then
        echo "[slurm] training n=2 AWGN fiber-Sinkhorn implant for seed=$CURRENT_SEED"
        "$PYTHON_BIN" -u scripts/train_turboae_awgn_implant.py \
          --device cuda \
          --seed "$CURRENT_SEED" \
          --n 2 \
          --ebno-db "$TURBOAE_EBNO_DB" \
          --rate "$TURBOAE_RATE" \
          --dataset-size "$CHANNEL_IMPLANT_DATASET_SIZE" \
          --eval-size "$CHANNEL_IMPLANT_EVAL_SIZE" \
          --batch-size "$CHANNEL_IMPLANT_BATCH_SIZE" \
          --epochs "$CHANNEL_IMPLANT_EPOCHS" \
          --swd-projections "$CHANNEL_IMPLANT_SWD_PROJECTIONS" \
          --drift-field fiber_sinkhorn \
          --conditioning-mode none \
          --sinkhorn-min-epsilon "$CHANNEL_IMPLANT_SINKHORN_MIN_EPSILON" \
          --sinkhorn-iterations "$CHANNEL_IMPLANT_SINKHORN_ITERATIONS" \
          --fiber-generated-samples "$CHANNEL_IMPLANT_FIBER_GENERATED_SAMPLES" \
          --fiber-positive-samples "$CHANNEL_IMPLANT_FIBER_POSITIVE_SAMPLES" \
          --fiber-reference-samples "$CHANNEL_IMPLANT_FIBER_REFERENCE_SAMPLES" \
          --anchor-metrics \
          --anchor-count "$CHANNEL_IMPLANT_ANCHOR_COUNT" \
          --anchor-samples "$CHANNEL_IMPLANT_ANCHOR_SAMPLES" \
          --anchor-swd-projections "$CHANNEL_IMPLANT_ANCHOR_SWD_PROJECTIONS" \
          --save-dir "$IMPLANT_DIR/checkpoints" \
          --out "$IMPLANT_DIR/summary.json"
      fi
    fi

    if [[ ! -f "$IMPLANT_CHECKPOINT" ]]; then
      echo "[slurm] missing checkpoint implant: $IMPLANT_CHECKPOINT" >&2
      exit 1
    fi
  fi

  CMD=(
    "$PYTHON_BIN" -u scripts/run_turboae_long_block_suite.py
    --lengths "$TURBOAE_LENGTHS"
    --modes "$TURBOAE_MODES"
    --checkpoint "$IMPLANT_CHECKPOINT"
    --channel "$TURBOAE_CHANNEL"
    --model-type "$TURBOAE_MODEL_TYPE"
    --turbo-root "$TURBO_ROOT"
    --device cuda
    --seed "$CURRENT_SEED"
    --epochs "$TURBOAE_EPOCHS"
    --batch-size "$TURBOAE_BATCH_SIZE"
    --dec-bs-fac "$TURBOAE_DEC_BS_FAC"
    --enc-micro-batch-size "$TURBOAE_ENC_MICRO_BATCH_SIZE"
    --dec-micro-batch-size "$TURBOAE_DEC_MICRO_BATCH_SIZE"
    --sample-size "$TURBOAE_SAMPLE_SIZE"
    --eval-num-blocks "$TURBOAE_EVAL_NUM_BLOCKS"
    --eval-batches "$TURBOAE_EVAL_BATCHES"
    --eval-every "$TURBOAE_EVAL_EVERY"
    --save-every "$TURBOAE_SAVE_EVERY"
    --learning-rate "$TURBOAE_LEARNING_RATE"
    --weight-decay "$TURBOAE_WEIGHT_DECAY"
    --grad-clip-norm "$TURBOAE_GRAD_CLIP_NORM"
    --ebno-db "$TURBOAE_EBNO_DB"
    --rate "$TURBOAE_RATE"
    --decoder-ebno-offset-low "$TURBOAE_DECODER_EBNO_OFFSET_LOW"
    --decoder-ebno-offset-high "$TURBOAE_DECODER_EBNO_OFFSET_HIGH"
    --lr-plateau-patience-evals "$TURBOAE_LR_PLATEAU_PATIENCE_EVALS"
    --lr-plateau-factor "$TURBOAE_LR_PLATEAU_FACTOR"
    --lr-plateau-max-reductions "$TURBOAE_LR_PLATEAU_MAX_REDUCTIONS"
    --lr-plateau-min-lr "$TURBOAE_LR_PLATEAU_MIN_LR"
    --out-dir "$SEED_DIR"
  )

  if is_enabled "$TURBOAE_ALLOW_TF32"; then
    CMD+=(--allow-tf32)
  fi
  if is_enabled "$TURBOAE_AMP"; then
    CMD+=(--amp --amp-dtype "$TURBOAE_AMP_DTYPE")
  fi
  if is_enabled "$TURBOAE_COMPILE_MODELS"; then
    CMD+=(--compile-models)
  fi
  if is_enabled "$TURBOAE_ENABLE_GPU_WARMUP"; then
    CMD+=(--enable-gpu-warmup)
  fi

  echo "[slurm] running TurboAE suite for seed=$CURRENT_SEED"
  "${CMD[@]}"
done
