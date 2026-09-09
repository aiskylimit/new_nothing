#!/usr/bin/env bash
set -e

PROJECT_DIR="$(pwd)"
DOWNLOAD_ROOT=/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main
DOWNLOAD_DATA_DIR="${DOWNLOAD_ROOT}/train_data"
DATA_DIR="${PROJECT_DIR}/train_data"

source /mnt/local/uvenvs/vlm-distillation/bin/activate

# Copy metadata into the relative train_data tree. cmp also safely handles
# the case where the download directory and code directory are the same.
mkdir -p "${DATA_DIR}/ocr_vqa"
if ! cmp -s "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" "${DATA_DIR}/llava_v1_5_mix665k.json"; then
  cp -f "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" "${DATA_DIR}/llava_v1_5_mix665k.json"
fi
if ! cmp -s "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" "${DATA_DIR}/ocr_vqa/dataset.json"; then
  cp -f "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" "${DATA_DIR}/ocr_vqa/dataset.json"
fi

# Training uses IMAGE_DIR=train_data. The image archives are expected to have
# already been extracted under this directory, as in project_commands.sh.
export CUDA_VISIBLE_DEVICES=4,5,6,7
bash script_train/run_baseline_2.sh
