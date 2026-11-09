#!/bin/bash
# End-to-end SFT(Label) and SFT(Long-CoT) reproduction on Qwen2.5-7B-Instruct
# and Qwen3-8B. Runs are serial and have isolated output directories.
#
# Usage:
#   bash project_commands.sh
#   bash project_commands.sh [all|env|data|train|eval] [all|qwen25_label|qwen25_longcot|qwen3_label|qwen3_longcot]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

STAGE="${1:-all}"
TARGET="${2:-qwen25_longcot}"
ALL_TARGETS="qwen25_label qwen25_longcot qwen3_label qwen3_longcot"

ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/palign_sft}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$ASSET_ROOT/datasets/s1K-1.1}"
EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-1000}"
export PALIGN_SFT_ASSET_ROOT="$ASSET_ROOT"
export PALIGN_SFT_DATA_DIR="$DATA_DIR"

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

NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29340}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-33}"
PER_DEVICE_BS="${PER_DEVICE_BS:-11}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1000}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ROOT/../palign/data/raw}"
EVAL_N="${EVAL_N:-3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
PYTHON="${PYTHON:-python}"

EVAL_ARGS=()
if [ "${EVAL_OVERWRITE:-0}" = "1" ]; then EVAL_ARGS+=(--overwrite); fi

usage() {
  echo "usage: bash project_commands.sh [all|env|data|train|eval] [all|qwen25_label|qwen25_longcot|qwen3_label|qwen3_longcot]" >&2
}

validate_target() {
  case "$1" in
    all|qwen25_label|qwen25_longcot|qwen3_label|qwen3_longcot) ;;
    *)
      echo "unknown target: $1" >&2
      usage
      exit 1
      ;;
  esac
}

load_target() {
  case "$1" in
    qwen25_label)
      TRAIN_CONFIG="configs/qwen25_7b_label_sft.yaml"
      EXPORT_CONFIG="configs/qwen25_7b_label_export.yaml"
      MODEL_DIR="$ASSET_ROOT/models/Qwen2.5-7B-Instruct"
      RUN_DIR="output/qwen25-7b-label"
      ;;
    qwen25_longcot)
      TRAIN_CONFIG="configs/qwen25_7b_longcot_sft.yaml"
      EXPORT_CONFIG="configs/qwen25_7b_longcot_export.yaml"
      MODEL_DIR="$ASSET_ROOT/models/Qwen2.5-7B-Instruct"
      RUN_DIR="output/qwen25-7b-longcot"
      ;;
    qwen3_label)
      TRAIN_CONFIG="configs/qwen3_8b_label_sft.yaml"
      EXPORT_CONFIG="configs/qwen3_8b_label_export.yaml"
      MODEL_DIR="$ASSET_ROOT/models/Qwen3-8B"
      RUN_DIR="output/qwen3-8b-label"
      ;;
    qwen3_longcot)
      TRAIN_CONFIG="configs/qwen3_8b_longcot_sft.yaml"
      EXPORT_CONFIG="configs/qwen3_8b_longcot_export.yaml"
      MODEL_DIR="$ASSET_ROOT/models/Qwen3-8B"
      RUN_DIR="output/qwen3-8b-longcot"
      ;;
    *)
      echo "internal error: unsupported target $1" >&2
      exit 1
      ;;
  esac
}

for_targets() {
  local function_name="$1"
  local name
  if [ "$TARGET" = "all" ]; then
    for name in $ALL_TARGETS; do
      "$function_name" "$name"
    done
  else
    "$function_name" "$TARGET"
  fi
}

cmd_env() {
  "$PYTHON" -c "import datasets, jsonlines, math_verify, peft, torch, transformers, vllm, yaml; import llamafactory; print('env ok')"
}

cmd_data() {
  mkdir -p "$DATA_DIR/raw" output/log
  if [ ! -e "$DATA_DIR/dataset_info.json" ] || \
    [ ! "$ROOT/data/dataset_info.json" -ef "$DATA_DIR/dataset_info.json" ]; then
    cp "$ROOT/data/dataset_info.json" "$DATA_DIR/dataset_info.json"
  fi
  "$PYTHON" src/prepare_s1k.py \
    --source "$TRAIN_SOURCE" \
    --output-dir "$DATA_DIR" \
    --expected-rows "$EXPECTED_TRAIN_ROWS"

  "$PYTHON" src/fetch_eval.py
}

cmd_train_one() {
  local name="$1"
  load_target "$name"
  if [ "$NNODES" -ne 1 ]; then
    echo "project_commands.sh supports single-node orchestration only (NNODES must be 1)" >&2
    exit 1
  fi
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "missing model checkpoint: $MODEL_DIR" >&2
    exit 1
  fi

  local world_size=$((NPROC_PER_NODE * NNODES))
  local batch_denominator=$((PER_DEVICE_BS * world_size))
  if [ "$batch_denominator" -le 0 ] || [ "$EFFECTIVE_BATCH" -lt "$batch_denominator" ] || [ $((EFFECTIVE_BATCH % batch_denominator)) -ne 0 ]; then
    echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH must be divisible by PER_DEVICE_BS*WORLD_SIZE=$batch_denominator" >&2
    exit 1
  fi
  local grad_accum=$((EFFECTIVE_BATCH / batch_denominator))
  mkdir -p "$RUN_DIR" output/log
  echo "[$name] SFT model=$MODEL_DIR nproc=$NPROC_PER_NODE grad_accum=$grad_accum effective_batch=$EFFECTIVE_BATCH"
  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
    src/train.py "$TRAIN_CONFIG" \
    model_name_or_path="$MODEL_DIR" \
    dataset_dir="$DATA_DIR" \
    output_dir="$RUN_DIR/lora" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    gradient_accumulation_steps="$grad_accum" seed="$TRAIN_SEED" data_seed="$TRAIN_SEED"

  echo "[$name] merging LoRA adapter"
  llamafactory-cli export "$EXPORT_CONFIG" \
    model_name_or_path="$MODEL_DIR" \
    adapter_name_or_path="$RUN_DIR/lora" \
    export_dir="$RUN_DIR/merged"
}

cmd_eval_one() {
  local name="$1"
  load_target "$name"
  local eval_model="$RUN_DIR/merged"
  if [ -n "${MODEL:-}" ]; then
    eval_model="$MODEL"
  fi
  if [ ! -f "$eval_model/config.json" ]; then
    echo "missing evaluation model: $eval_model" >&2
    exit 1
  fi

  local result_dir="$RUN_DIR/result"
  mkdir -p "$result_dir"
  echo "[$name] evaluating model=$eval_model"
  "$EVAL_PYTHON" "$ROOT/../palign/src/run_eval.py" \
    --model "$eval_model" --data-dir "$EVAL_DATA_DIR" --result-dir "$result_dir" \
    --n "$EVAL_N" --batch-size "$EVAL_BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
}

case "$STAGE" in
  env)
    cmd_env
    ;;
  data)
    cmd_data
    ;;
  train)
    validate_target "$TARGET"
    cmd_data
    for_targets cmd_train_one
    ;;
  eval)
    validate_target "$TARGET"
    if [ "$TARGET" = "all" ] && [ -n "${MODEL:-}" ]; then
      echo "MODEL override may only be used with a single eval target" >&2
      exit 1
    fi
    cmd_data
    for_targets cmd_eval_one
    ;;
  all)
    validate_target "$TARGET"
    if [ "$TARGET" = "all" ] && [ -n "${MODEL:-}" ]; then
      echo "MODEL override may only be used with a single target" >&2
      exit 1
    fi
    cmd_env
    cmd_data
    for_targets cmd_train_one
    for_targets cmd_eval_one
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    usage
    exit 1
    ;;
esac
