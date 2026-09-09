#!/usr/bin/env bash
set -euo pipefail

# Keep the GPU list supplied by the launch command, for example:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/qwen/train_rvd_qwen2.5_14b_to_1.5b.sh
# GPU_IDS is retained as a backward-compatible fallback.
SELECTED_GPU_IDS="${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0,1,2,3}}"
IFS=, read -r -a GPUS <<< "$SELECTED_GPU_IDS"
if (( ${#GPUS[@]} == 0 )); then
    printf 'No GPU was selected. Set CUDA_VISIBLE_DEVICES to a comma-separated GPU list.\n' >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for GPU_ID in "${GPUS[@]}"; do
    if [[ -z "$GPU_ID" || "$GPU_ID" == *[[:space:]]* ]]; then
        printf 'Invalid GPU list: %s\n' "$SELECTED_GPU_IDS" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[$GPU_ID]:-}" ]]; then
        printf 'GPU %s is listed more than once in: %s\n' "$GPU_ID" "$SELECTED_GPU_IDS" >&2
        exit 2
    fi
    SEEN_GPUS[$GPU_ID]=1
done
export GPU_IDS="$(IFS=,; printf '%s' "${GPUS[*]}")"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# Model
BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
CKPT_NAME="qwen2.5-1.5B-Instruct"
CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER_CKPT_NAME="qwen2.5-14B-Instruct"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"

DATA_DIR="${DATA_DIR:-${BASE_PATH}/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct}"

# Hyperparameters
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
GRAD_ACC="${GRAD_ACC:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
SEED="${SEED:-10}"


STEP_SEPARATOR=$'\n\n'
STEP_POOLING="${STEP_POOLING:-mean}"
MAGNITUDE_NORMALIZATION="${MAGNITUDE_NORMALIZATION:-zscore}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
MAG_WEIGHT="${MAG_WEIGHT:-1.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-1.0}"
LOGIT_WEIGHT="${LOGIT_WEIGHT:-1.0}"
DISTILL_TOP_K="${DISTILL_TOP_K:-32}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"

SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/${CKPT_NAME}-rvd-topk/${STEP_POOLING}_k${DISTILL_TOP_K}_bs${BATCH_SIZE}_ga${GRAD_ACC}_lr${LR}_seed${SEED}}"
DS_CONFIG="${DS_CONFIG:-${BASE_PATH}/configs/deepspeed/ds_config_bf16.json}"

OPTS=()

# Model
OPTS+=(--base-path "$BASE_PATH")
OPTS+=(--model-path "$CKPT" --model-type qwen)
OPTS+=(--teacher-model-path "$TEACHER_CKPT" --teacher-model-type qwen)
OPTS+=(--ckpt-name "$CKPT_NAME" --teacher-ckpt-name "$TEACHER_CKPT_NAME")
OPTS+=(--n-gpu "$GPUS_PER_NODE" --n-nodes "$NNODES")
OPTS+=(--bf16)

# Data
OPTS+=(--data-dir "$DATA_DIR" --json-data)
OPTS+=(--num-workers "$NUM_WORKERS" --dev-num -1)

# Hyperparameters
OPTS+=(--lr "$LR" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE")
OPTS+=(--gradient-accumulation-steps "$GRAD_ACC" --gradient-checkpointing)
OPTS+=(--warmup-iters 0 --lr-decay-style cosine)
OPTS+=(--weight-decay 1e-2 --clip-grad 1.0 --epochs "$EPOCHS")

# Top-K requires matching teacher/student prediction positions and length limits.
OPTS+=(--max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH")
OPTS+=(--t-max-length "$MAX_LENGTH" --t-max-prompt-length "$MAX_PROMPT_LENGTH")

# Runtime: save and evaluate after every epoch.
OPTS+=(--do-train --do-valid --eval-gen)
OPTS+=(--save-interval -1 --eval-interval -1 --log-interval 10)
OPTS+=(--save "$SAVE_PATH" --seed "$SEED")

# DeepSpeed BF16 + ZeRO-1; the repository config is shared with the runtime.
OPTS+=(--deepspeed --deepspeed_config "$DS_CONFIG")

# Distillation
OPTS+=(--type rvd)
OPTS+=(--step-separator "$STEP_SEPARATOR" --step-pooling "$STEP_POOLING")
OPTS+=(--magnitude-normalization "$MAGNITUDE_NORMALIZATION" --eps 1e-6)
OPTS+=(--ce-weight "$CE_WEIGHT" --mag-weight "$MAG_WEIGHT")
OPTS+=(--gram-weight "$GRAM_WEIGHT" --logit-weight "$LOGIT_WEIGHT")
OPTS+=(--distill-top-k "$DISTILL_TOP_K" --distill-temperature "$DISTILL_TEMPERATURE")

# Validation generation
OPTS+=(--do-sample --top-k 0 --top-p 1.0 --temperature 1.0 --num-beams 1)

# Student LoRA
OPTS+=(--peft lora --peft-lora-r 16 --peft-lora-alpha 128 --peft-lora-dropout 0.05)

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export CODE_BASE=HF

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${BASE_PATH}/finetune.py" "${OPTS[@]}" "$@")
printf 'CUDA_VISIBLE_DEVICES=%s\nPYTHONPATH=%s\n' "$CUDA_VISIBLE_DEVICES" "$PYTHONPATH"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

mkdir -p -- "$SAVE_PATH"
exec "${CMD[@]}"
