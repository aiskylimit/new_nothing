#!/usr/bin/env bash
# End-to-end SEGD cls pipeline for offline server (aiskylimit).
# Prerequisite: prefetch data + models via segd_distillation/download.txt
# (no wget/hf download in this script).

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/local/aiskylimit_new_nothing/segd_distillation}"
UVENV="${UVENV:-/mnt/local/uvenvs/segd-distillation}"
MODELS_ROOT="${MODELS_ROOT:-${PROJECT_ROOT}/models}"
DATASETS_ROOT="${DATASETS_ROOT:-${PROJECT_ROOT}/datasets}"

cd "$PROJECT_ROOT"
source "${UVENV}/bin/activate"

# Offline server: local mirrors from download.txt
export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/FastVLM-0.5B}"
export TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-${MODELS_ROOT}/B3_Qwen2_2B}"
export DATASET_NAME="${DATASET_NAME:-vlm2vec_train/MMEB-train}"
export IMAGE_DIR="${IMAGE_DIR:-vlm2vec_train/MMEB-train}"

python fix_lib.py

# #
# # 3. Unzip the dataset (zips prefetched to ${DATASETS_ROOT}/ by download.txt)
# #
mkdir -p vlm2vec_train/MMEB-train/images
mkdir -p eval_images
unzip -o "${DATASETS_ROOT}/ImageNet_1K.zip" -d ./vlm2vec_train/MMEB-train/images/
unzip -o "${DATASETS_ROOT}/HatefulMemes.zip" -d ./vlm2vec_train/MMEB-train/images/
unzip -o "${DATASETS_ROOT}/VOC2007.zip" -d ./vlm2vec_train/MMEB-train/images/
unzip -o "${DATASETS_ROOT}/N24News.zip" -d ./vlm2vec_train/MMEB-train/images/
unzip -o "${DATASETS_ROOT}/SUN397.zip" -d ./vlm2vec_train/MMEB-train/images/
unzip -o "${DATASETS_ROOT}/images.zip" -d ./eval_images/




CUDA_VISIBLE_DEVICES=1 bash scripts/cls/run_segd_cls.sh &
wait


# =========================
# 8. Eval
# =========================

CUDA_VISIBLE_DEVICES=1 bash scripts/cls/eval_segd_cls.sh &
wait

# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
