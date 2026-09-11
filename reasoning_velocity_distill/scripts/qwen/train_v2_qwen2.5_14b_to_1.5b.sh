#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0,1}}}"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
GPUS_PER_NODE=${#GPUS[@]}
NNODES="${NNODES:-1}"
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# Model and data
BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"
CKPT_NAME="qwen2.5-1.5B-Instruct"
TEACHER_CKPT_NAME="qwen2.5-14B-Instruct"
CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"
DATA_DIR="${DATA_DIR:-$BASE_PATH/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct}"
# Optional prepared JSONL; otherwise train data must already contain context.
PRIVILEGED_DATA_PATH="${PRIVILEGED_DATA_PATH:-}"
DS_CONFIG="${DS_CONFIG:-$BASE_PATH/configs/deepspeed/ds_config_bf16.json}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-1536}"
T_MAX_LENGTH="${T_MAX_LENGTH:-$((T_MAX_PROMPT_LENGTH + MAX_LENGTH))}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEV_NUM="${DEV_NUM:-512}"
SEED="${SEED:-10}"

# Adaptive OFF -> privileged, with randomly mixed ON-policy batches.
KD_LOSS="${KD_LOSS:-sfkl}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
KD_RATIO="${KD_RATIO:-0.5}"
MAG_WEIGHT="${MAG_WEIGHT:-1.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-1.0}"
DISTILL_TOP_K="${DISTILL_TOP_K:-512}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
STEP_SEPARATOR="${STEP_SEPARATOR:-$'\n\n'}"
STEP_POOLING="${STEP_POOLING:-mean}"
MAGNITUDE_NORMALIZATION="${MAGNITUDE_NORMALIZATION:-zscore}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/${CKPT_NAME}-v2/adaptive_${KD_LOSS}_k${DISTILL_TOP_K}_bs${BATCH_SIZE}_ga${GRAD_ACC}_lr${LR}_seed${SEED}}"

OPTS=()

# Model and data
OPTS+=(--base-path "$BASE_PATH")
OPTS+=(--model-path "$CKPT" --model-type qwen --ckpt-name "$CKPT_NAME")
OPTS+=(--teacher-model-path "$TEACHER_CKPT" --teacher-model-type qwen --teacher-ckpt-name "$TEACHER_CKPT_NAME")
OPTS+=(--n-gpu "$GPUS_PER_NODE" --n-nodes "$NNODES" --bf16)
OPTS+=(--data-dir "$DATA_DIR" --json-data --num-workers "$NUM_WORKERS" --dev-num "$DEV_NUM")
if [[ -n "$PRIVILEGED_DATA_PATH" ]]; then
    OPTS+=(--privileged-data-path "$PRIVILEGED_DATA_PATH")
fi

# Training: 'wrmup_cosine' is the scheduler name supported by finetune_v2.
OPTS+=(--lr "$LR" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE")
OPTS+=(--gradient-accumulation-steps "$GRAD_ACC" --gradient-checkpointing)
OPTS+=(--lr-decay-style wrmup_cosine --warmup-ratio "$WARMUP_RATIO")
OPTS+=(--weight-decay 1e-2 --clip-grad 1.0 --epochs "$EPOCHS")
OPTS+=(--max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH")
OPTS+=(--t-max-length "$T_MAX_LENGTH" --t-max-prompt-length "$T_MAX_PROMPT_LENGTH")

# Geometry applies only to OFF batches; the scheduler controls mode transitions.
OPTS+=(--type kd --distill-mode off_policy --off-policy-geometry)
OPTS+=(--adaptive-on-policy --do-sample --progress-signal loss)
OPTS+=(--rho-min "${RHO_MIN:-0.1}" --rho-max "${RHO_MAX:-0.8}" --rho-increment "${RHO_INCREMENT:-0.05}")
OPTS+=(--privileged-trajectory canonical --privileged-context-field context)
OPTS+=(--kd-loss "$KD_LOSS" --kd-ratio "$KD_RATIO")
OPTS+=(--skew-alpha "$SKEW_ALPHA")
OPTS+=(--mag-weight "$MAG_WEIGHT" --gram-weight "$GRAM_WEIGHT")
OPTS+=(--distill-top-k "$DISTILL_TOP_K" --distill-temperature "$DISTILL_TEMPERATURE")
OPTS+=(--step-separator "$STEP_SEPARATOR" --step-pooling "$STEP_POOLING")
OPTS+=(--magnitude-normalization "$MAGNITUDE_NORMALIZATION" --eps 1e-6)
OPTS+=(--peft lora --peft-lora-r "$LORA_R" --peft-lora-alpha "$LORA_ALPHA" --peft-lora-dropout "$LORA_DROPOUT")

# Save each epoch; evaluate loss every 100 optimizer steps for the scheduler.
OPTS+=(--do-train --do-valid)
OPTS+=(--save-interval -1 --eval-interval "$EVAL_INTERVAL" --log-interval 10 --mid-log-num 0)
OPTS+=(--save "$SAVE_PATH" --seed "$SEED" --seed-data "$SEED" --seed-lm "$SEED")
OPTS+=(--top-k 0 --top-p 1.0 --temperature 1.0 --repetition-penalty 1.0 --num-beams 1)
OPTS+=(--deepspeed --deepspeed_config "$DS_CONFIG")

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CODE_BASE=HF

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "$BASE_PATH/finetune_v2.py" "${OPTS[@]}" "$@")
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

mkdir -p -- "$SAVE_PATH"
exec "${CMD[@]}"
