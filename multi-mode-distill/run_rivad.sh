#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
export CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract-v2}"
export DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1}" SEED="${SEED:-42}"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# RIVAD adaptive routing with geometry disabled for this comparison.
export GEOMETRY=0 CKA=0 ADAPTIVE_MODE_SET=all
export MAG_WEIGHT="${MAG_WEIGHT:-2.0}" GRAM_WEIGHT="${GRAM_WEIGHT:-2.0}"
export KD_RATIO="${KD_RATIO:-0.5}" EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/rebuttal/adaptive_seed${SEED}}"

# DRY_RUN=1 prints the training command.
# RUN_EVAL=0 trains and saves all four checkpoints without downstream eval.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
    source "$VENV_PATH/bin/activate"
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
