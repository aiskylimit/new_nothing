#!/usr/bin/env bash
set -e

PROJECT_DIR="$(pwd)"
DOWNLOAD_ROOT=/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main
DOWNLOAD_DATA_DIR="${DOWNLOAD_ROOT}/train_data"
DATA_DIR="${PROJECT_DIR}/train_data"

source /mnt/local/uvenvs/vlm-distillation-main/bin/activate

# Copy metadata into the relative train_data tree.  cmp also safely handles
# the case where the download directory and code directory are the same.
mkdir -p "${DATA_DIR}/ocr_vqa"
if ! cmp -s "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" "${DATA_DIR}/llava_v1_5_mix665k.json"; then
  cp -f "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" "${DATA_DIR}/llava_v1_5_mix665k.json"
fi
if ! cmp -s "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" "${DATA_DIR}/ocr_vqa/dataset.json"; then
  cp -f "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" "${DATA_DIR}/ocr_vqa/dataset.json"
fi

# Training uses the relative IMAGE_DIR=train_data.  The JSON image values start
# with coco/, gqa/, ocr_vqa/, textvqa/, and vg/.
unzip -q -o "${DOWNLOAD_DATA_DIR}/coco/train2017.zip" -d "${DATA_DIR}/coco"
unzip -q -o "${DOWNLOAD_DATA_DIR}/gqa/images.zip" -d "${DATA_DIR}/gqa"
unzip -q -o "${DOWNLOAD_DATA_DIR}/textvqa/train_val_images.zip" -d "${DATA_DIR}/textvqa"
unzip -q -o "${DOWNLOAD_DATA_DIR}/ocr_vqa/ocr_vqa_images.zip" -d "${DATA_DIR}/ocr_vqa"
unzip -q -o "${DOWNLOAD_DATA_DIR}/vg/images.zip" -d "${DATA_DIR}/vg"
unzip -q -o "${DOWNLOAD_DATA_DIR}/vg/images2.zip" -d "${DATA_DIR}/vg"

#bash download_datatrain.sh

export CUDA_VISIBLE_DEVICES=4,5
bash script_train/run_baseline.sh
