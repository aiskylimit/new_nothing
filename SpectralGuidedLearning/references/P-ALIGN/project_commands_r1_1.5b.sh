#!/bin/bash
# P-ALIGN on DeepSeek-R1-Distill-Qwen-1.5B, full finetuning (no LoRA / no merge).
# Train in non-thinking mode (empty <think></think> in the prompt) and eval the same way.
# No Hub upload, no git push.
#
# Usage:
#   bash project_commands_r1_1.5b.sh            # data check + train + eval
#   bash project_commands_r1_1.5b.sh env
#   bash project_commands_r1_1.5b.sh data
#   bash project_commands_r1_1.5b.sh train
#   bash project_commands_r1_1.5b.sh eval       # needs trained weights (or MODEL=...)
#
# Eval base checkpoint only (skip train):
#   MODEL=/mnt/local/_models/aiskylimit_new_nothingnew_2/DeepSeek-R1-Distill-Qwen-1.5B \
#     RESULT_DIR=output/result_r1_1.5b_base bash project_commands_r1_1.5b.sh eval
#
# Multi-GPU with DeepSpeed ZeRO-2 (shards optimizer state; use if a single GPU OOMs):
#   NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 DEEPSPEED=configs/ds_z2.json bash project_commands_r1_1.5b.sh train

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/P-ALIGN}"
MODEL_PATH="${MODEL_PATH:-/mnt/local/_models/aiskylimit_new_nothingnew_2/DeepSeek-R1-Distill-Qwen-1.5B}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
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
# Make CUDA_VISIBLE_DEVICES indices match nvidia-smi (PCI order) instead of FASTEST_FIRST.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG=configs/r1_distill_qwen_1.5b_palign_full_sft.yaml
OUTPUT_DIR="${OUTPUT_DIR:-output/palign-r1-distill-qwen-1.5b-full}"
MODEL="${MODEL:-$OUTPUT_DIR}"
RESULT_DIR="${RESULT_DIR:-output/result_r1_1.5b}"
EVAL_OUT="${EVAL_OUT:-output/eval_results_r1_1.5b.txt}"
DEEPSPEED="${DEEPSPEED:-}"
# vLLM reserves this fraction of the whole GPU at eval; lower it (e.g. 0.3) if the GPU is shared
GPU_MEM="${GPU_MEM:-0.8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29332}"
EFFECTIVE_BATCH=32
PER_DEVICE_BS=1
STAGE="${1:-all}"
BENCHES=(aime25 aime24 amc12 math500)

cmd_env() {
  python -c "import torch, transformers, llamafactory, vllm; print('env ok')"
  if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "missing $MODEL_PATH (download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, see download.txt)" >&2
    exit 1
  fi
}

cmd_data() {
  mkdir -p "$DATA_DIR/raw" output/log "$RESULT_DIR"
  if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ] && [ -f "$DATA_DIR/palign_sft_qwen2.5-7b.json.gz" ]; then
    gunzip -kc "$DATA_DIR/palign_sft_qwen2.5-7b.json.gz" > "$DATA_DIR/palign_sft_qwen2.5-7b.json"
  fi
  if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ]; then
    echo "missing local train file $DATA_DIR/palign_sft_qwen2.5-7b.json" >&2
    exit 1
  fi
  for b in "${BENCHES[@]}"; do
    if [ ! -f "$DATA_DIR/raw/$b.jsonl" ]; then
      python src/fetch_eval.py
      break
    fi
  done
}

cmd_train() {
  mkdir -p output/log
  WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
  GRAD_ACCUM=$((EFFECTIVE_BATCH / (PER_DEVICE_BS * WORLD_SIZE)))
  echo "full SFT nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH} deepspeed=${DEEPSPEED:-none}"
  # --standalone: cluster pods export PET_RDZV_* which otherwise hangs single-node rendezvous
  if [ "$NNODES" -eq 1 ]; then
    LAUNCH_ARGS=(--standalone --nproc_per_node "$NPROC_PER_NODE")
  else
    LAUNCH_ARGS=(--nproc_per_node "$NPROC_PER_NODE" --nnodes "$NNODES" --node_rank "$RANK"
                 --rdzv_backend static --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT")
  fi
  EXTRA_ARGS=()
  if [ -n "$DEEPSPEED" ]; then
    EXTRA_ARGS+=(deepspeed="$DEEPSPEED")
  fi
  # num_train_epochs=3: the final save in $OUTPUT_DIR is the epoch-3 model that eval uses.
  torchrun "${LAUNCH_ARGS[@]}" \
    src/train.py "$CONFIG" \
    model_name_or_path="$MODEL_PATH" \
    output_dir="$OUTPUT_DIR" \
    gradient_accumulation_steps="$GRAD_ACCUM" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

cmd_eval() {
  mkdir -p "$RESULT_DIR"
  IN=(); OUT=()
  for b in "${BENCHES[@]}"; do
    IN+=("$DATA_DIR/raw/$b.jsonl")
    OUT+=("$RESULT_DIR/$b.jsonl")
  done
  # --force_empty_think: thinking off, prompt ends with "<｜Assistant｜><think>\n\n</think>\n\n" as in training
  python src/test.py \
    --model "$MODEL" \
    --input_files "${IN[@]}" \
    --output_files "${OUT[@]}" \
    --batch_size 1000 \
    --n 3 \
    --temperature 0.6 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --max_tokens 4096 \
    --gpu_memory_utilization "$GPU_MEM" \
    --force_empty_think
  for f in "${OUT[@]}"; do
    python src/evaluation.py --input_path "$f" --output_path "${f%.jsonl}_scored.jsonl"
  done
  python src/report.py --result_dir "$RESULT_DIR" --out "$EVAL_OUT"
  echo "wrote $EVAL_OUT"
}

case "$STAGE" in
  env) cmd_env ;;
  data) cmd_data ;;
  train) cmd_data; cmd_train ;;
  eval) cmd_data; cmd_eval ;;
  all) cmd_env; cmd_data; cmd_train; cmd_eval ;;
  *)
    echo "usage: bash project_commands_r1_1.5b.sh [all|env|data|train|eval]" >&2
    exit 1
    ;;
esac
