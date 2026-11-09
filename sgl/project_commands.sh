#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MASK_DIR="${MASK_DIR:-artifacts/qwen2.5-7b}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen2.5-7b-sgl}"
EVALUATION_DIR="${EVALUATION_DIR:-outputs/qwen2.5-7b-sgl}"

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_NEW_TOKENS=4096
TRAIN_SEED="${TRAIN_SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
EVAL_N="${EVAL_N:-3}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON_BIN}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ROOT_DIR/../palign/data/raw}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-33}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-11}"

EVAL_ARGS=()
if [ "${EVAL_OVERWRITE:-0}" = "1" ]; then EVAL_ARGS+=(--overwrite); fi

echo "[1/4] Installing the project and development dependencies"
"$PYTHON_BIN" -m pip install -e ".[dev]"

echo "[2/4] Building spectral masks"
"$PYTHON_BIN" -m accelerate.commands.launch --multi_gpu --num_processes "$NPROC_PER_NODE" -m sgl.build_masks \
  --model-name-or-path "$MODEL_NAME" \
  --output-dir "$MASK_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --max-length "$MAX_LENGTH" \
  --rank-threshold 0.95 \
  --selection-threshold 0.8 \
  --svd-device cuda \
  --svd-driver gesvd \
  --dtype bfloat16 \
  --attn-implementation sdpa

echo "[3/4] Training the LoRA adapter"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" -m sgl.training \
  --mask-dir "$MASK_DIR" \
  --output-dir "$CHECKPOINT_DIR" \
  --model-name-or-path "$MODEL_NAME" \
  --seed "$TRAIN_SEED" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-ratio 0.1 \
  --gradient-checkpointing \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules \
    q_proj \
    k_proj \
    v_proj \
    o_proj \
    gate_proj \
    up_proj \
    down_proj

echo "[4/4] Evaluating the trained adapter"
"$EVAL_PYTHON" "$ROOT_DIR/../palign/src/run_eval.py" \
  --model "$CHECKPOINT_DIR" --result-dir "$EVALUATION_DIR" \
  --data-dir "$EVAL_DATA_DIR" --n "$EVAL_N" \
  --max-tokens "$MAX_NEW_TOKENS" --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}

echo "Pipeline completed successfully."
echo "Masks:      $MASK_DIR"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Evaluation: $EVALUATION_DIR"
