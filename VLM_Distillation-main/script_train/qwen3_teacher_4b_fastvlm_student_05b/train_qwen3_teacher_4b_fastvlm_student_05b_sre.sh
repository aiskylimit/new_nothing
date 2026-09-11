#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

STUDENT_MODEL="${STUDENT_MODEL:-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/KamilaMila/FastVLM-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen3-VL-4B-Instruct}"
DATA_PATH="${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json"
IMAGE_DIR="${PROJECT_DIR}/train_data"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3_teacher_4b_fastvlm_student_05b_sre"

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
  --lora_r 128 \
  --lora_alpha 256 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 100 \
  --dataloader_num_workers 2 \
  --train_sampling_strategy group_by_length \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --kd_loss_type "sre" \
  --sre_selective_attention true \
  --sre_use_projector true \
  --teacher_layer_mapping -3 -1 \
  --student_layer_mapping -3 -1 \
  --sre_alpha 0.5 \
  --sre_p 1.0 \
  --sre_span_loss_weight 1.0 \
  --sre_geom_loss_weight 3.0 \
  --sre_logit_loss_weight 1.0 \
  --sre_temperature 2.0 \
  --sre_skew_loss_weight 1.0 \
  --sre_skew_lambda 0.01 \
  --projector_lr 5e-4
