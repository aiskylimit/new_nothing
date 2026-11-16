#!/usr/bin/env bash
set -euo pipefail

# Evaluate the final checkpoint of adaptive seed 42 and the predetermined run.
export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
export CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}"
export EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill-eval}/bin/python}"

RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/rebuttal}"
RUN_NAME="${RUN_NAME:-e2-bs8-lr0.0001-G4-N4-NN1-kd0.5-lora-16-128-0.05}"

for RUN in "predetermined_seed${PREDETERMINED_SEED:-10}"; do
    RUN_DIR="$RESULTS_ROOT/$RUN/$RUN_NAME"
    FINAL_STEP="$("$EVAL_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_iters"])' "$RUN_DIR/args.json")"
    LORA_PATH="$RUN_DIR/$FINAL_STEP"
    printf '\n[eval %s 100%%] step=%s, checkpoint=%s\n' "$RUN" "$FINAL_STEP" "$LORA_PATH"
    LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" SAVE_PATH="$RUN_DIR" \
        EVAL_OUTPUT_DIR="$LORA_PATH/evaluation" \
        bash scripts/eval/eval.sh run
done
