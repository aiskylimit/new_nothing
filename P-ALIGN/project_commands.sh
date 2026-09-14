#!/bin/bash
# All run commands for P-ALIGN on Qwen2.5-7B-Instruct.
# No Hub upload, no git push. Train JSON is local: data/palign_sft_qwen2.5-7b.json
#
# Usage:
#   bash project_commands.sh            # env + data check + train + merge + eval
#   bash project_commands.sh env
#   bash project_commands.sh data
#   bash project_commands.sh train
#   bash project_commands.sh eval       # needs merged weights (or MODEL=...)
#
# Eval base checkpoint only (skip train/merge):
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash project_commands.sh eval

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MERGED="${MERGED:-output/palign-qwen2.5-7b-instruct-lora-merged}"
MODEL="${MODEL:-$MERGED}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29330}"
EFFECTIVE_BATCH=32
PER_DEVICE_BS=1
STAGE="${1:-all}"

cmd_env() {
  python -m pip install -r requirements.txt
}

cmd_data() {
  mkdir -p data/raw output/log output/result
  if [ ! -f data/palign_sft_qwen2.5-7b.json ]; then
    echo "missing local train file data/palign_sft_qwen2.5-7b.json" >&2
    exit 1
  fi
  for f in data/raw/aime25.jsonl data/raw/aime24.jsonl data/raw/amc12.jsonl data/raw/math500.jsonl; do
    if [ ! -f "$f" ]; then
      python src/fetch_eval.py
      break
    fi
  done
}

cmd_train() {
  mkdir -p output/log
  WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
  GRAD_ACCUM=$((EFFECTIVE_BATCH / (PER_DEVICE_BS * WORLD_SIZE)))
  echo "SFT nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
  torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    src/train.py configs/qwen2.5_7b_palign_sft.yaml \
    gradient_accumulation_steps="$GRAD_ACCUM"
  llamafactory-cli export configs/qwen2.5_7b_palign_export.yaml
}

cmd_eval() {
  mkdir -p output/result
  python src/test.py \
    --model "$MODEL" \
    --input_files data/raw/aime25.jsonl data/raw/aime24.jsonl data/raw/amc12.jsonl data/raw/math500.jsonl \
    --output_files output/result/aime25.jsonl output/result/aime24.jsonl output/result/amc12.jsonl output/result/math500.jsonl \
    --batch_size 1000 \
    --n 3 \
    --temperature 0.6 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --max_tokens 4096
  for f in output/result/aime25.jsonl output/result/aime24.jsonl output/result/amc12.jsonl output/result/math500.jsonl; do
    python src/evaluation.py --input_path "$f" --output_path "${f%.jsonl}_scored.jsonl"
  done
  python src/report.py --out output/eval_results.txt
  echo "wrote output/eval_results.txt"
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
