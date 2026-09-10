#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"

VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-4,5}"

export CKPT="${CKPT:-$BASE_PATH/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$BASE_PATH/models/Qwen2.5_14B-Instruct}"
export DATA_DIR="${DATA_DIR:-$BASE_PATH/processed_data/ultraInteract/models/$(basename -- "$CKPT")}"
export RAW_DATA="${RAW_DATA:-$BASE_PATH/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
export SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-rvd}"
export BATCH_SIZE="${BATCH_SIZE:-8}" GRAD_ACC="${GRAD_ACC:-2}"
export EVAL_VENV_PATH="${EVAL_VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill-eval}"
export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export SEED="${SEED:-10}"
export TOKENIZERS_PARALLELISM=false

# # 1. Validate assets downloaded from download.txt.
# [[ -f "$RAW_DATA" ]] || { printf 'Training data not found: %s\n' "$RAW_DATA" >&2; exit 1; }
# [[ -d "$CKPT" ]] || { printf 'Student model not found: %s\n' "$CKPT" >&2; exit 1; }
# [[ -d "$TEACHER_CKPT" ]] || { printf 'Teacher model not found: %s\n' "$TEACHER_CKPT" >&2; exit 1; }

# if [[ "${RUN_EVAL:-1}" == "1" ]]; then
#     bash scripts/eval/eval.sh check
# fi

# # 2. Build the local train/validation JSONL files.
# if [[ "${RUN_PREPROCESS:-1}" == "1" ]]; then
#     PYTHONPATH="$BASE_PATH" python tools/process_data_ultraInteract.py \
#         --base-path "$BASE_PATH" \
#         --data-dir "$RAW_DATA" \
#         --processed-data-dir "$BASE_PATH/processed_data/ultraInteract" \
#         --model-path "$CKPT" \
#         --model-type qwen \
#         --max-length "$MAX_LENGTH" \
#         --max-prompt-length "$MAX_PROMPT_LENGTH" \
#         --data-process-workers "${DATA_PROCESS_WORKERS:-8}" \
#         --dev-num "${DEV_NUM:-200}" \
#         --seed "$SEED"
# fi

# # 3. Train and save the final checkpoint under SAVE_PATH/<step>.
# if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
#     CUDA_VISIBLE_DEVICES=4,5 bash scripts/qwen/train_rvd_qwen2.5_14b_to_1.5b.sh &
#     TRAIN_PID=$!
# fi

# # 4. Evaluate the final checkpoint using only local benchmark mirrors.
# if [[ "${RUN_EVAL:-1}" == "1" ]]; then
#     # The training command runs in the background. Wait for it before loading
#     # the final checkpoint; wait also propagates a training failure.
#     if [[ -n "$TRAIN_PID" ]]; then
#         wait "$TRAIN_PID"
#     fi
#     CUDA_VISIBLE_DEVICES=4,5 bash scripts/eval/eval.sh run
# fi

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    bash scripts/eval/eval.sh check
fi

CUDA_VISIBLE_DEVICES=4,5 \
LORA_PATH="$PWD/results/qwen2.5-1.5B-Instruct-rvd/7455" \
bash scripts/eval/eval.sh run