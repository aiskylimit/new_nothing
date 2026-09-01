#!/usr/bin/env bash

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${_SCRIPT_DIR}"
ASSETS_ROOT="/mnt/local/aiskylimit_new_nothing/segd_distillation"
UVENV="${PROJECT_ROOT}/.venv"
export CUDA_VISIBLE_DEVICES=1
MODELS_ROOT="${ASSETS_ROOT}/models"
DATASETS_ROOT="${ASSETS_ROOT}/datasets"
TRAIN_DATA_ROOT="${ASSETS_ROOT}/vlm2vec_train/MMEB-train"

cd "$PROJECT_ROOT"
source "${UVENV}/bin/activate"

export MODEL_NAME="${MODELS_ROOT}/FastVLM-0.5B"
export TEACHER_MODEL_NAME="${MODELS_ROOT}/B3_Qwen2_2B"
export EXP_NAME="FastVLM-0.5B_segd_eos_cls_full"
export PERCENT_DATA=1.0

python fix_lib.py

TRAIN_IMAGES_DIR="${TRAIN_DATA_ROOT}/images"
EVAL_IMAGES_DIR="${PROJECT_ROOT}/eval_images"
mkdir -p "$TRAIN_IMAGES_DIR" "$EVAL_IMAGES_DIR"

for subset in ImageNet_1K HatefulMemes VOC2007 N24News SUN397; do
    unzip -q -o "${DATASETS_ROOT}/${subset}.zip" -d "$TRAIN_IMAGES_DIR"
done

unzip -q -o "${DATASETS_ROOT}/images.zip" -d "$EVAL_IMAGES_DIR"

export IMAGE_DIR="${TRAIN_DATA_ROOT}"
bash scripts/cls/run_segd_cls.sh

EVAL_MODEL="training/${EXP_NAME}/checkpoint-final"
MODEL="$EVAL_MODEL" bash scripts/cls/eval_segd_cls.sh

python json_filter.py ./MMEB-eval_outputs ./MMEB-evaloutputs-json --overwrite
