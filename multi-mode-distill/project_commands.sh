#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

export CKPT="${CKPT:-$BASE_PATH/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$BASE_PATH/models/Qwen2.5_14B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$BASE_PATH/processed_data/ultraInteract-v2}"

# Use the preprocessor's path resolution unless an existing DATA_DIR is supplied.
if [[ -z "${DATA_DIR:-}" ]]; then
    DATA_DIR="$(python -c 'import sys; from tools.process_data_ultraInteract import resolve_processed_data_dir; print(resolve_processed_data_dir(*sys.argv[1:]))' \
        "$PROCESSED_DATA_ROOT" "$CKPT" "$BASE_PATH")"
fi
export DATA_DIR
export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}" SEED="${SEED:-10}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
# Reserve the student sequence plus only the extra context budget.
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    exit 1
fi

# 1. Train synchronously using the existing processed data.
printf '\n[1/2] Train v2: dual adaptive OFF/self-distill/ON exposure\n'
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES=4,5,6,7 FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

# 2. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run
