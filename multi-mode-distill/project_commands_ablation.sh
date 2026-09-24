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
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract-v2}"

# The old preprocessor stores absolute model paths under models/<model name>.
if [[ -z "${DATA_DIR:-}" ]]; then
    DATA_DIR="$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")"
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


MODE_CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$MODE_CHECKPOINT_FILE"' EXIT
RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/adaptive_mode_ablation}"
read -r -a ADAPTIVE_ABLATIONS <<< "${ADAPTIVE_ABLATIONS:-on_self off_self}"

for ADAPTIVE_MODE_SET in "${ADAPTIVE_ABLATIONS[@]}"; do
    case "$ADAPTIVE_MODE_SET" in
        on_self|off_self) ;;
        *) printf 'Unsupported adaptive ablation: %s\n' "$ADAPTIVE_MODE_SET" >&2; exit 2 ;;
    esac

    printf '\n[%s 1/2] Train adaptive Qwen with modes: %s\n' \
        "$ADAPTIVE_MODE_SET" "$ADAPTIVE_MODE_SET"
    : > "$MODE_CHECKPOINT_FILE"
    CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
        DATA_DIR="$DATA_DIR" ADAPTIVE_MODE_SET="$ADAPTIVE_MODE_SET" \
        SAVE_PATH="$RESULTS_ROOT/$ADAPTIVE_MODE_SET" \
        FINAL_CHECKPOINT_FILE="$MODE_CHECKPOINT_FILE" \
        bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

    MODE_LORA_PATH="$(cat "$MODE_CHECKPOINT_FILE")"
    [[ -f "$MODE_LORA_PATH/adapter_config.json" ]] || {
        printf 'Final LoRA checkpoint missing: %s\n' "$MODE_LORA_PATH" >&2
        exit 1
    }
    printf '\n[%s 2/2] Evaluate checkpoint: %s\n' \
        "$ADAPTIVE_MODE_SET" "$MODE_LORA_PATH"
    CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
        LORA_PATH="$MODE_LORA_PATH" MODEL_PATH="$CKPT" \
        SAVE_PATH="$(dirname -- "$MODE_LORA_PATH")" \
        EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
        bash scripts/eval/eval.sh run
done
