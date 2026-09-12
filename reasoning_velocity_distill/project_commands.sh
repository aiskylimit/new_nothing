#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"
export EVAL_VENV_PATH="${EVAL_VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill-eval}"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Models and source data
export CKPT="${CKPT:-$BASE_PATH/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$BASE_PATH/models/Qwen2.5_14B-Instruct}"
RAW_DATA="${RAW_DATA:-$BASE_PATH/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
CONTEXT_DATA_PATH="${CONTEXT_DATA_PATH:-${RAW_DATA%.jsonl}_with_context.jsonl}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$BASE_PATH/processed_data/ultraInteract-v2}"

# Use the preprocessor's own path resolution for local and Hugging Face models.
DATA_DIR="$(python -c 'import sys; from tools.process_data_ultraInteract import resolve_processed_data_dir; print(resolve_processed_data_dir(*sys.argv[1:]))' \
    "$PROCESSED_DATA_ROOT" "$CKPT" "$BASE_PATH")"
export DATA_DIR
export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}" SEED="${SEED:-10}"
CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-1024}"
CONTEXT_MAX_PROMPT_LENGTH="${CONTEXT_MAX_PROMPT_LENGTH:-8192}"
# Context generation has its own budget. Privileged training uses the same
# bounded student response, with additional teacher space for prompt + context.
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
# A short student prompt can leave nearly MAX_LENGTH tokens for the response;
# reserve that full budget, rather than MAX_LENGTH - MAX_PROMPT_LENGTH.
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((T_MAX_PROMPT_LENGTH + MAX_LENGTH))}"

# 1. Generate context for the FULL raw dataset, before splitting.
printf '\n[1/4] Generate context for full dataset: %s\n' "$CONTEXT_DATA_PATH"
if [[ ! -f "$CONTEXT_DATA_PATH" ]]; then
    CUDA_VISIBLE_DEVICES=4,5 python prepare_privileged_data.py \
        --data-dir "$RAW_DATA" --output "$CONTEXT_DATA_PATH" \
        --teacher-model-path "$TEACHER_CKPT" --device-map auto --dtype bfloat16 \
        --batch-size "${CONTEXT_BATCH_SIZE:-32}" \
        --max-new-tokens "$CONTEXT_MAX_NEW_TOKENS" \
        --max-prompt-length "$CONTEXT_MAX_PROMPT_LENGTH" \
        --max-length "$((CONTEXT_MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))" \
        --privileged-context-field context --seed "$SEED"
fi

# 2. Preprocess and split; each record retains its generated context.
printf '\n[2/4] Preprocess dataset with context: %s\n' "$DATA_DIR"
if [[ ! -f "$DATA_DIR/.context-preprocessed" \
      || "$CONTEXT_DATA_PATH" -nt "$DATA_DIR/.context-preprocessed" \
      || "$BASE_PATH/tools/process_data_ultraInteract.py" -nt "$DATA_DIR/.context-preprocessed" \
      || ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/valid.jsonl" ]]; then
    rm -f "$DATA_DIR/.context-preprocessed"
    python tools/process_data_ultraInteract.py \
        --base-path "$BASE_PATH" --data-dir "$CONTEXT_DATA_PATH" \
        --processed-data-dir "$PROCESSED_DATA_ROOT" \
        --model-path "$CKPT" --model-type qwen \
        --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --data-process-workers "${DATA_PROCESS_WORKERS:-8}" \
        --dev-num "$DEV_NUM" --seed "$SEED"
    touch "$DATA_DIR/.context-preprocessed"
fi

# 3. Train synchronously; context is already inside train.jsonl.
printf '\n[3/4] Train v2: adaptive OFF -> privileged + random ON-policy\n'
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES=4,5 PRIVILEGED_DATA_PATH= FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

# 4. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[4/4] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run
