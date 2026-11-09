#!/bin/bash
# All run commands for P-ALIGN on Qwen2.5-7B-Instruct.
# No Hub upload, no git push. Train JSON is local: data/palign_sft_qwen2.5-7b.json
#
# Usage:
#   bash project_commands.sh            # data check + train + merge + eval
#   bash project_commands.sh env
#   bash project_commands.sh data
#   bash project_commands.sh train
#   bash project_commands.sh eval       # needs merged weights (or MODEL=...)
#
# Eval base checkpoint only (skip train/merge):
#   MODEL=/mnt/local/aiskylimit_new_nothing/P-ALIGN/models/Qwen2.5-7B-Instruct bash project_commands.sh eval
#
# Tiny 0.5B flow check (does not replace this script):
#   bash project_commands_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/P-ALIGN}"
MODEL_PATH="${MODEL_PATH:-$ASSET_ROOT/models/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR/raw}"
export PALIGN_ASSET_ROOT="$ASSET_ROOT"
export PALIGN_DATA_DIR="$DATA_DIR"
if [ -f "${PALIGN_VENV:-/mnt/local/uvenvs/p-align}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PALIGN_VENV:-/mnt/local/uvenvs/p-align}/bin/activate"
elif [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
else
  echo "no python env (PALIGN_VENV or /mnt/local/uvenvs/p-align or /venv/main)" >&2
  exit 1
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

MERGED="${MERGED:-output/palign-qwen25-7b-lora-merged}"
MODEL="${MODEL:-$MERGED}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29330}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-33}"
PER_DEVICE_BS="${PER_DEVICE_BS:-11}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_N="${EVAL_N:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
EVAL_ARGS=()
if [ "${EVAL_OVERWRITE:-0}" = "1" ]; then EVAL_ARGS+=(--overwrite); fi

STAGE="${1:-all}"

cmd_env() {
  python -c "import torch, transformers, llamafactory, vllm; print('env ok')"
}

cmd_data() {
  mkdir -p "$DATA_DIR/raw" output/log output/result
  if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ]; then
    echo "missing local train file $DATA_DIR/palign_sft_qwen2.5-7b.json" >&2
    exit 1
  fi
  for f in "$DATA_DIR/raw/aime25.jsonl" "$DATA_DIR/raw/aime24.jsonl" "$DATA_DIR/raw/amc12.jsonl" "$DATA_DIR/raw/math500.jsonl"; do
    if [ ! -f "$f" ]; then
      python src/fetch_eval.py
      break
    fi
  done
}

cmd_train() {
  mkdir -p output/log
  WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
  if [ "$NNODES" -ne 1 ]; then
    echo "single-node orchestration requires NNODES=1" >&2
    exit 1
  fi
  DENOM=$((PER_DEVICE_BS * WORLD_SIZE))
  if [ "$DENOM" -le 0 ] || [ "$EFFECTIVE_BATCH" -lt "$DENOM" ] || [ $((EFFECTIVE_BATCH % DENOM)) -ne 0 ]; then
    echo "EFFECTIVE_BATCH must be a positive multiple of PER_DEVICE_BS * WORLD_SIZE" >&2
    exit 1
  fi
  GRAD_ACCUM=$((EFFECTIVE_BATCH / DENOM))
  echo "SFT nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
    src/train.py configs/qwen25_7b_palign_sft.yaml \
    model_name_or_path="$MODEL_PATH" dataset_dir="$DATA_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    gradient_accumulation_steps="$GRAD_ACCUM" seed="$TRAIN_SEED" data_seed="$TRAIN_SEED"
  llamafactory-cli export configs/qwen25_7b_palign_export.yaml model_name_or_path="$MODEL_PATH" export_dir="$MERGED"
}

cmd_eval() {
  "$EVAL_PYTHON" src/run_eval.py --model "$MODEL" \
    --data-dir "$EVAL_DATA_DIR" --result-dir output/result \
    --n "$EVAL_N" --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
}

case "$STAGE" in
  env) cmd_env ;;
  data) cmd_data ;;
  train) cmd_data; cmd_train ;;
  eval) cmd_data; cmd_eval ;;
  all) cmd_env; cmd_data; cmd_train; cmd_eval ;;
  *)
    echo "usage: bash project_commands.sh [all|env|data|train|eval]" >&2
    exit 1
    ;;
esac
