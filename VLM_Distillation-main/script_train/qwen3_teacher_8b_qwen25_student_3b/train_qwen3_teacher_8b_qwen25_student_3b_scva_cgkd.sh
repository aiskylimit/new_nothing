#!/usr/bin/env bash
# SCVA + CGKD joint — the draft.pdf headline method, 8-GPU.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen3_teacher_8b_qwen25_student_3b_scva_cgkd}"
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
# source "${PROJECT_DIR}/script_train/_common.sh"

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
  --weight_decay 0.01 \
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
  --kd_loss_type scva_cgkd \
  --scva_n_clusters 16 \
  --scva_kmeans_iters 10 \
  --scva_min_vision_tokens 4 \
  --cgkd_temperature 1.0 \
  --scva_cgkd_ce_weight 0.7 \
  --scva_cgkd_lambda_v 0.3 \
  --scva_cgkd_lambda_g 0.3 \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
