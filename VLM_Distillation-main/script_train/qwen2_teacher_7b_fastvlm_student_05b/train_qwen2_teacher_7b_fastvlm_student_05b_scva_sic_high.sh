#!/usr/bin/env bash
# Joint SCVA + SIC, high decoder layers [0.7, 1.0].
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

MODEL_ROOT="${MODEL_ROOT:-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models}"
STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT}/KamilaMila/FastVLM-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen/Qwen2-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen2_teacher_7b_fastvlm_student_05b_scva_sic_high}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${RUN_NAME}}"

PERCENT_DATA="${PERCENT_DATA:-1.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
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
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 50 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --train_sampling_strategy group_by_length \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --seed 1337 \
  --kd_loss_type scva_sic \
  --scva_n_layer_pairs "${SCVA_N_LAYER_PAIRS:-4}" \
  --scva_layer_low_pct 0.7 \
  --scva_layer_high_pct 1.0 \
  --scva_min_vision_tokens "${SCVA_MIN_VISION_TOKENS:-4}" \
  --scva_spatial_weight "${SCVA_SPATIAL_WEIGHT:-0.1}" \
  --scva_dbscan_min_samples "${SCVA_DBSCAN_MIN_SAMPLES:-8}" \
  --scva_dbscan_eps_percentile "${SCVA_DBSCAN_EPS_PERCENTILE:-3.0}" \
  --scva_sic_ce_weight "${SCVA_SIC_CE_WEIGHT:-1.0}" \
  --scva_sic_lambda_v "${SCVA_SIC_LAMBDA_V:-0.3}" \
  --scva_sic_lambda_sic "${SCVA_SIC_LAMBDA_SIC:-0.3}" \
  --sic_num_regions "${SIC_NUM_REGIONS:-16}" \
  --sic_max_clusters "${SIC_MAX_CLUSTERS:-0}" \
  --sic_use_projector "${SIC_USE_PROJECTOR:-true}" \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
