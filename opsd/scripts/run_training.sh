#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: bash scripts/run_training.sh <sft|grpo|opsd> <4b|8b|olmo7b>}"
MODEL_SIZE="${2:?Usage: bash scripts/run_training.sh <sft|grpo|opsd> <4b|8b|olmo7b>}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/experiment_settings.env"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-${PROJECT_ROOT}/data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/accelerate.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-auto}"
if [[ "${MAIN_PROCESS_PORT}" == "auto" ]]; then
    MAIN_PROCESS_PORT="$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi
# Torchrun reads PET_* defaults before Accelerate overlays its launch arguments.
# Ignore cluster-injected rendezvous settings so the selected local port is used.
unset PET_RDZV_ENDPOINT PET_RDZV_ID PET_RDZV_BACKEND PET_RDZV_CONF
unset PET_MASTER_ADDR PET_MASTER_PORT PET_STANDALONE
NUM_PROCESSES="${NUM_PROCESSES:-${TRAIN_NUM_PROCESSES}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"

case "${MODEL_SIZE}" in
    4b)
        MODEL_PATH="${MODEL_ROOT}/Qwen3-4B"
        PER_DEVICE_BATCH="${PER_DEVICE_BATCH_4B}"
        GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION_4B}"
        OPSD_CLIP="${OPSD_CLIP_4B}"
        MODEL_TAG="qwen3_4b"
        ;;
    8b)
        MODEL_PATH="${MODEL_ROOT}/Qwen3-8B"
        PER_DEVICE_BATCH="${PER_DEVICE_BATCH_8B}"
        GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION_8B}"
        OPSD_CLIP="${OPSD_CLIP_8B}"
        MODEL_TAG="qwen3_8b"
        ;;
    olmo7b)
        MODEL_PATH="${MODEL_ROOT}/Olmo-3-7B-Think"
        PER_DEVICE_BATCH="${PER_DEVICE_BATCH_OLMO7B}"
        GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION_OLMO7B}"
        OPSD_CLIP="${OPSD_CLIP_OLMO7B}"
        MODEL_TAG="olmo3_7b_think"
        ;;
    *)
        echo "Unsupported model size: ${MODEL_SIZE}. Expected 4b, 8b, or olmo7b." >&2
        exit 2
        ;;
esac

if [[ "${METHOD}" == "grpo" ]]; then
    PER_DEVICE_BATCH="${GRPO_PER_DEVICE_BATCH}"
    GRADIENT_ACCUMULATION="${GRPO_GRADIENT_ACCUMULATION}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/train" ]]; then
    echo "Missing prepared training data: ${PREPARED_DATA_ROOT}/train" >&2
    exit 1
fi

ACTUAL_EFFECTIVE_BATCH=$((PER_DEVICE_BATCH * NUM_PROCESSES * GRADIENT_ACCUMULATION))
if [[ "${ACTUAL_EFFECTIVE_BATCH}" -ne "${EFFECTIVE_BATCH_SIZE}" ]]; then
    echo "Effective batch mismatch: ${PER_DEVICE_BATCH} x ${NUM_PROCESSES} x ${GRADIENT_ACCUMULATION} = ${ACTUAL_EFFECTIVE_BATCH}, expected ${EFFECTIVE_BATCH_SIZE}." >&2
    exit 1
fi

RUN_CONFIG="${METHOD}_${MODEL_TAG}_paper"
METHOD_OUTPUT_ROOT="${OUTPUT_ROOT}/${METHOD}"
mkdir -p "${METHOD_OUTPUT_ROOT}"

COMMON_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --local_dataset_path "${PREPARED_DATA_ROOT}/train"
    --output_dir "${METHOD_OUTPUT_ROOT}"
    --run_config "${RUN_CONFIG}"
    --learning_rate "${LEARNING_RATE}"
    --per_device_train_batch_size "${PER_DEVICE_BATCH}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}"
    --gradient_checkpointing
    --use_peft
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --attn_implementation sdpa
    --torch_dtype bfloat16
    --optim adamw_torch
    --save_strategy steps
    --logging_strategy steps
    --eval_strategy no
    --report_to none
    --seed "${SEED}"
)

echo "Method: ${METHOD}"
echo "Model: ${MODEL_PATH}"
echo "Batch: ${PER_DEVICE_BATCH} x ${NUM_PROCESSES} GPUs x ${GRADIENT_ACCUMULATION} accumulation = ${ACTUAL_EFFECTIVE_BATCH}"
echo "Rendezvous port: ${MAIN_PROCESS_PORT}"

case "${METHOD}" in
    sft)
        ENTRYPOINT="${PROJECT_ROOT}/sft_train.py"
        METHOD_ARGS=(
            --max_steps "${SFT_MAX_STEPS}"
            --max_length "${SFT_MAX_LENGTH}"
            --save_steps "${SFT_SAVE_STEPS}"
            --logging_steps 5
        )
        ;;
    grpo)
        ENTRYPOINT="${PROJECT_ROOT}/grpo_train.py"
        METHOD_ARGS=(
            --max_steps "${GRPO_MAX_STEPS}"
            --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}"
            --num_generations "${GRPO_NUM_GENERATIONS}"
            --num_iterations 2
            --temperature "${TRAIN_TEMPERATURE}"
            --top_p "${TRAIN_TOP_P}"
            --top_k "${TRAIN_TOP_K}"
            --beta "${GRPO_BETA}"
            --loss_type grpo
            --scale_rewards group
            --use_vllm
            --vllm_mode colocate
            --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
            --vllm_tensor_parallel_size 1
            --save_steps "${GRPO_SAVE_STEPS}"
            --logging_steps 10
        )
        ;;
    opsd)
        ENTRYPOINT="${PROJECT_ROOT}/opsd_train.py"
        METHOD_ARGS=(
            --max_steps "${OPSD_MAX_STEPS}"
            --max_grad_norm 0.1
            --max_length "${OPSD_MAX_LENGTH}"
            --max_completion_length "${OPSD_MAX_COMPLETION_LENGTH}"
            --temperature "${TRAIN_TEMPERATURE}"
            --top_p "${TRAIN_TOP_P}"
            --top_k "${TRAIN_TOP_K}"
            --beta "${OPSD_BETA}"
            --lmbda 1
            --fixed_teacher
            --top_k_loss 0
            --jsd_token_clip "${OPSD_CLIP}"
            --student_thinking "${STUDENT_THINKING:-False}"
            --teacher_thinking "${TEACHER_THINKING:-True}"
            --use_vllm
            --vllm_mode colocate
            --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
            --vllm_tensor_parallel_size 1
            --save_steps "${OPSD_SAVE_STEPS}"
            --logging_steps 2
        )
        ;;
    *)
        echo "Unsupported method: ${METHOD}. Expected sft, grpo, or opsd." >&2
        exit 2
        ;;
esac

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${ENTRYPOINT}" \
    "${COMMON_ARGS[@]}" \
    "${METHOD_ARGS[@]}"
