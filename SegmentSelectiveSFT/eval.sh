#!/usr/bin/env bash
# Eval rebuttal dung chung vLLM, prompt P-ALIGN va math-verify.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${EVAL_DATA_DIR:-$ROOT/../palign/data/raw}"
RESULT_DIR="${RESULT_DIR:-$ROOT/Eval/outputs_rebuttal}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
N="${EVAL_N:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0,1,2}}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --base) MODEL="$BASE_MODEL"; shift ;;
    --data-dir|--data_dir) DATA_DIR="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --tag) RESULT_DIR="$ROOT/Eval/outputs_$2"; shift 2 ;;
    --n|--n-sampling) N="$2"; shift 2 ;;
    --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --tasks) read -r -a TASKS <<< "$2"; EXTRA+=(--benchmarks "${TASKS[@]}"); shift 2 ;;
    --overwrite|--score-only) EXTRA+=("$1"); shift ;;
    --max-tokens|--max-model-len|--temperature|--top-p|--repetition-penalty|--seed|--tensor-parallel-size)
      EXTRA+=("$1" "$2"); shift 2 ;;
    -h|--help) exec "$EVAL_PYTHON" "$ROOT/../palign/src/run_eval.py" --help ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$MODEL" ]]; then
  MODEL="$ROOT/SelectiveSFT/checkpoints/$(basename "$BASE_MODEL")_epoch3_lr5e-5_len32768_lora/checkpoint-final"
fi
exec "$EVAL_PYTHON" "$ROOT/../palign/src/run_eval.py" \
  --model "$MODEL" --data-dir "$DATA_DIR" --result-dir "$RESULT_DIR" \
  --n "$N" ${EXTRA[@]+"${EXTRA[@]}"}
