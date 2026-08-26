#!/usr/bin/env bash
# SRE-only (text-span alignment + geometry + shared-vocab KL).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen25_teacher_7b_qwen2_student_2b_sre}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"
PERCENT_DATA="${PERCENT_DATA:-1.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

cd "${PROJECT_DIR}"
[[ -x "${TORCHRUN}" ]] || TORCHRUN="torchrun"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/script_train/_common.sh"

# Qwen2-VL-2B has 28 transformer layers (indices 0..27); we map last layer to last.
"${TORCHRUN}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --data_path "${DATA_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --percent_data "${PERCENT_DATA}" \
  --lora true \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --logging_steps 50 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --seed 1337 \
  --kd_loss_type sre \
  --sre_use_projector true \
  --teacher_layer_mapping 27 \
  --student_layer_mapping 27 \
  --sre_alpha 0.5 \
  --sre_p 1.0 \
  --sre_span_loss_weight 1.0 \
  --sre_geom_loss_weight 50 \
  --sre_logit_loss_weight 1.0 \
  --sre_temperature 2.0 \
  --projector_lr 1e-4 \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
