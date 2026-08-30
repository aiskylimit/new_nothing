#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

STUDENT_MODEL="${STUDENT_MODEL:-/workspace/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-3B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-/workspace/VLM_Distillation-main/models/Qwen/Qwen3-VL-8B-Instruct}"
DATA_PATH="${DATA_PATH:-train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-train_data}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3_teacher_8b_qwen25_student_3b_dwa_kd"
PROJECTOR_CONFIG="${PROJECT_DIR}/config/dwa_kd_projectors.json"

MASTER_PORT="${MASTER_PORT:-29501}"
STUDENT_HIDDEN_DIM="${STUDENT_HIDDEN_DIM:-2048}"
TEACHER_HIDDEN_DIM="${TEACHER_HIDDEN_DIM:-4096}"

cd "${PROJECT_DIR}"


torchrun \
  --nproc_per_node gpu \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --student_hidden_dim "${STUDENT_HIDDEN_DIM}" \
  --teacher_hidden_dim "${TEACHER_HIDDEN_DIM}" \
  --projector_config_path "${PROJECTOR_CONFIG}" \
  --data_path "${DATA_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --percent_data 1.0 \
  --lora true \
  --lora_r 128 \
  --lora_alpha 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --projector_lr 5e-5 \
  --weight_decay 0.01 \
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
  --kd_loss_type "dwa_kd" \
  --kd_objective "forward_kl" \
  --ce_rate 1.0 \
  --kd_rate 1.0 \
  --dtw_rate 0.1 \
  --kd_temperature 1.0 \
  --teacher_temperature 1.0 \
  --kd_warmup_steps 300 \
  --dtw_gamma 2.0 \
  --dtw_gamma_start 2.0 \
  --dtw_gamma_end 0.8 \
  --dtw_gamma_steps 3570 \
  --dtw_band_width 5 \
  --dtw_band_source "cma"
