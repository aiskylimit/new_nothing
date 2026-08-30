#!/usr/bin/env bash
# CGKD-only (single-method draft.pdf component) — 8-GPU.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen25_teacher_7b_qwen2_student_2b_cgkd}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"
PERCENT_DATA="${PERCENT_DATA:-1.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

cd "${PROJECT_DIR}"


torchrun \
  --nproc_per_node gpu \
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
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 50 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --seed 1337 \
  --kd_loss_type cgkd \
  --cgkd_alpha 0.5 \
  --cgkd_weight 1.0 \
  --cgkd_temperature 1.0 \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
