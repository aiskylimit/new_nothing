#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

export CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
GEMMA_CKPT="${GEMMA_CKPT:-$ASSET_ROOT/models/google_gemma-2-2b-it}"
GEMMA_TEACHER_CKPT="${GEMMA_TEACHER_CKPT:-$ASSET_ROOT/models/google_gemma-2-9b-it}"
GEMMA_RAW_DATA="${GEMMA_RAW_DATA:-$ASSET_ROOT/data/raw/google/gemma-2-9b-it/generated_train.jsonl}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract-v2}"
# The old preprocessor stores absolute model paths under models/<model name>.
GEMMA_DATA_DIR="${GEMMA_DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$GEMMA_CKPT")}"
QWEN_DATA_DIR="${QWEN_DATA_DIR:-${DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")}}"

export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}" SEED="${SEED:-10}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
# Reserve the student sequence plus only the extra context budget.
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

# Process Gemma data once before both Gemma training modes.
printf '\n[process] Gemma data: %s\n' "$GEMMA_RAW_DATA"
# qwen here selects uint32 token storage; --model-path still loads Gemma's tokenizer.
python tools/process_data_ultraInteract.py \
    --base-path "$BASE_PATH" --data-dir "$GEMMA_RAW_DATA" \
    --processed-data-dir "$PROCESSED_DATA_ROOT" \
    --model-path "$GEMMA_CKPT" --model-type qwen \
    --data-process-workers "${DATA_PROCESS_WORKERS:-8}" \
    --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --dev-num "$DEV_NUM" --seed "$SEED"

#Train gemmma-no ce loss
# 1. Train synchronously using the existing processed data.
printf '\n[1/2] Train v2: dual adaptive OFF/self-distill/ON exposure\n'
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES=4,5,6,7 CKPT="$GEMMA_CKPT" TEACHER_CKPT="$GEMMA_TEACHER_CKPT" \
    DATA_DIR="$GEMMA_DATA_DIR" KD_RATIO=1.0 \
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/gemma/train_gemma2_9b_to_2b.sh --disable-lm-loss "$@"

# 2. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$GEMMA_CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run

#train gemma with ce loss
# 1. Train synchronously using the existing processed data.
printf '\n[1/2] Train v2: dual adaptive OFF/self-distill/ON exposure\n'
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES=4,5,6,7 CKPT="$GEMMA_CKPT" TEACHER_CKPT="$GEMMA_TEACHER_CKPT" \
    DATA_DIR="$GEMMA_DATA_DIR" KD_RATIO="${CE_KD_RATIO:-0.5}" \
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/gemma/train_gemma2_9b_to_2b.sh "$@"

# 2. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$GEMMA_CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run

#Train qwen-no ce loss
if [[ ! -s "$QWEN_DATA_DIR/train.jsonl" || ( ! -s "$QWEN_DATA_DIR/valid.jsonl" && ! -s "$QWEN_DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$QWEN_DATA_DIR" >&2
    exit 1
fi
# 1. Train synchronously using the existing processed data.
printf '\n[1/2] Train v2: dual adaptive OFF/self-distill/ON exposure\n'
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" KD_RATIO=1.0 FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh --disable-lm-loss "$@"

# 2. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run

#train qwen with ce loss
# 1. Train synchronously using the existing processed data.
printf '\n[1/2] Train v2: dual adaptive OFF/self-distill/ON exposure\n'
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" KD_RATIO="${CE_KD_RATIO:-0.5}" FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

# 2. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run
