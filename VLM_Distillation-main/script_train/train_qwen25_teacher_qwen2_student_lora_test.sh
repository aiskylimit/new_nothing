#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json"
IMAGE_DIR="${PROJECT_DIR}/train_data"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen2_vl_2b_lora_distill_test"

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
  --percent_data 1.0 \
  --lora true \
  --lora_r 8 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 100 \
  --dataloader_num_workers 2 \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --kd_loss_type "sre" 
