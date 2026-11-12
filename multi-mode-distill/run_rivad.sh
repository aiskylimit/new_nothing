#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
export CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract-v2}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}" SEED="${SEED:-42}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
QWEN_RAW_DATA="${QWEN_RAW_DATA:-$ASSET_ROOT/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}"
# Keep the train/valid split fixed when changing the training seed.
PREPROCESS_SEED="${PREPROCESS_SEED:-10}"

# RIVAD adaptive routing with geometry disabled for this comparison.
export GEOMETRY=0 CKA=0 ADAPTIVE_MODE_SET=all
export MAG_WEIGHT="${MAG_WEIGHT:-2.0}" GRAM_WEIGHT="${GRAM_WEIGHT:-2.0}"
export KD_RATIO="${KD_RATIO:-0.5}" EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/rebuttal/adaptive_seed${SEED}}"

# DRY_RUN=1 prints preprocessing/training commands without running them.
# RUN_EVAL=0 trains and saves all four checkpoints without downstream eval.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
    source "$VENV_PATH/bin/activate"
fi

export DATA_DIR="$(python -c 'import sys; from tools.process_data_ultraInteract import resolve_processed_data_dir; print(resolve_processed_data_dir(*sys.argv[1:]))' \
    "$PROCESSED_DATA_ROOT" "$CKPT" "$BASE_PATH")"

# Process Qwen data before training.
PROCESS_CMD=(
    python tools/process_data_ultraInteract.py
    --base-path "$BASE_PATH" --data-dir "$QWEN_RAW_DATA"
    --processed-data-dir "$PROCESSED_DATA_ROOT"
    --model-path "$CKPT" --model-type qwen
    --data-process-workers "${DATA_PROCESS_WORKERS:-8}"
    --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH"
    --dev-num "$DEV_NUM" --seed "$PREPROCESS_SEED"
)
printf '\n[process] Raw data: %s\n[process] Output: %s\n[process] Split seed: %s\n' \
    "$QWEN_RAW_DATA" "$DATA_DIR" "$PREPROCESS_SEED"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'Command: '
    printf '%q ' "${PROCESS_CMD[@]}"
    printf '\n'
else
    "${PROCESS_CMD[@]}"
    printf '[process] Finished. Training will read: %s\n' "$DATA_DIR"
fi

export EXPOSURE_SCHEDULE=adaptive
export EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}"

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
TIMING_FILE="$SAVE_PATH/timing.log"
TOTAL_START=$SECONDS

printf '\n[train] RIVAD, seed=%s, output=%s\n' "$SEED" "$SAVE_PATH"
TRAIN_START=$SECONDS
FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
    --save-interval 0 --save-fractions 0.25 0.5 0.75 1.0 \
    --track-experiment --experiment-log-interval 100 "$@"

if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
printf '[train] Finished in %ss\n' "$((SECONDS - TRAIN_START))" | tee "$TIMING_FILE"
if [[ "${RUN_EVAL:-1}" == 0 ]]; then exit 0; fi

FINAL_CHECKPOINT="$(cat "$CHECKPOINT_FILE")"
RUN_DIR="$(dirname -- "$FINAL_CHECKPOINT")"
FINAL_STEP="$(basename -- "$FINAL_CHECKPOINT")"

# Evaluate the four milestones only after training has finished.
for PERCENT in 25 50 75 100; do
    STEP=$(((FINAL_STEP * PERCENT + 99) / 100))
    LORA_PATH="$RUN_DIR/$STEP"
    printf '\n[eval %s%%] step=%s, checkpoint=%s\n' "$PERCENT" "$STEP" "$LORA_PATH"
    EVAL_START=$SECONDS
    LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" SAVE_PATH="$RUN_DIR" \
        EVAL_OUTPUT_DIR="$LORA_PATH/evaluation" \
        bash scripts/eval/eval.sh run
    printf '[eval %s%%] Finished in %ss\n' "$PERCENT" "$((SECONDS - EVAL_START))" | tee -a "$TIMING_FILE"
done

printf '\n[done] Total time: %ss\n' "$((SECONDS - TOTAL_START))" | tee -a "$TIMING_FILE"
