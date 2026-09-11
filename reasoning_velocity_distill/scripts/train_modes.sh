#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-off_policy}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    off_policy|on_policy|privileged) ;;
    *) printf 'Expected off_policy, on_policy, or privileged\n' >&2; exit 2 ;;
esac
BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_PATH"

# Activate the project's existing environment before invoking this script.
CKPT="${CKPT:-$BASE_PATH/models/Qwen2.5_1.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-$BASE_PATH/models/Qwen2.5_14B-Instruct}"
DATA_DIR="${DATA_DIR:-$BASE_PATH/tests/fixtures/mode_sanity}"
SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/mode_sanity/$MODE}"
DS_CONFIG="${DS_CONFIG:-$BASE_PATH/configs/deepspeed/ds_config_bf16.json}"

OPTS=(
    --model-path "$CKPT" --teacher-model-path "$TEACHER_CKPT"
    --model-type qwen --teacher-model-type qwen
    --data-dir "$DATA_DIR" --save "$SAVE_PATH"
    --type kd --distill-mode "$MODE" --kd-loss fkl --kd-ratio 0.5
    --distill-top-k "${DISTILL_TOP_K:-32}" --distill-temperature "${DISTILL_TEMPERATURE:-1.0}"
    --do-train --total-iters "${TOTAL_ITERS:-2}"
    --batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
    --max-length 256 --max-prompt-length 128
    --t-max-length 512 --t-max-prompt-length 384
    --lr 1e-5 --lr-decay-style constant --warmup-iters 0
    --log-interval 1 --eval-interval 0 --save-interval 0
    --peft lora --peft-lora-r 8 --peft-lora-alpha 16
    --deepspeed --deepspeed_config "$DS_CONFIG"
)
if [[ "$MODE" == on_policy ]]; then OPTS+=(--do-sample); fi
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
CMD=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}"
     "$BASE_PATH/finetune_v2.py" "${OPTS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
exec "${CMD[@]}"
