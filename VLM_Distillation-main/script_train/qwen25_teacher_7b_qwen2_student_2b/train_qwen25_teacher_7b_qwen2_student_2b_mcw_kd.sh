#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json"
IMAGE_DIR="${PROJECT_DIR}/train_data"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen25_teacher_7b_qwen2_student_2b_mcw_kd"
PROJECTOR_CONFIG="${PROJECT_DIR}/config/mcw_kd_projectors.json"

MASTER_PORT="${MASTER_PORT:-29501}"
STUDENT_HIDDEN_DIM="${STUDENT_HIDDEN_DIM:-1536}"
TEACHER_HIDDEN_DIM="${TEACHER_HIDDEN_DIM:-3584}"

cd "${PROJECT_DIR}"


torchrun \
  --nproc_per_node gpu \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --student_hidden_dim "${STUDENT_HIDDEN_DIM}" \
  --teacher_hidden_dim "${TEACHER_HIDDEN_DIM}" \
  --proj_dim 512 \
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
  --logging_steps 10 \
  --dataloader_num_workers 2 \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --kd_loss_type "mcw_kd" \
  --kd_objective "forward_kl" \
  --ce_rate 1.0 \
  --kd_rate 5.0 \
  --kd_temperature 1.0 \
  --teacher_temperature 1.0 \
  --top_k_vocab 400 \
  --mcw_tau_seq 2.0 \
  --mcw_window_size 4 \
  --mcw_ot_logits_rate 1.0 \
  --mcw_ot_hidden_rate 1.0 \
  --mcw_sinkhorn_alpha 0.1 \
  --mcw_sinkhorn_iter 100
