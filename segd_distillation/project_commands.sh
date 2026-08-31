#!/usr/bin/env bash

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Override for offline aiskylimit: PROJECT_ROOT=/mnt/local/aiskylimit_new_nothing/segd_distillation
PROJECT_ROOT="${PROJECT_ROOT:-${_SCRIPT_DIR}}"
UVENV="${UVENV:-${PROJECT_ROOT}/.venv}"
export CUDA_VISIBLE_DEVICES=1
MODELS_ROOT="${MODELS_ROOT:-${PROJECT_ROOT}/models}"
DATASETS_ROOT="${DATASETS_ROOT:-${PROJECT_ROOT}/datasets}"

cd "$PROJECT_ROOT"
source "${UVENV}/bin/activate"

# Local models/images prefetched by download.txt.
export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/FastVLM-0.5B}"
export TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-${MODELS_ROOT}/B3_Qwen2_2B}"
export EXP_NAME="${EXP_NAME:-FastVLM-0.5B_segd_eos_cls_full}"
export PERCENT_DATA="${PERCENT_DATA:-1.0}"

python fix_lib.py

# Extract prefetched images once.
TRAIN_IMAGES_DIR="${PROJECT_ROOT}/vlm2vec_train/MMEB-train/images"
EVAL_IMAGES_DIR="${PROJECT_ROOT}/eval_images"
mkdir -p "$TRAIN_IMAGES_DIR" "$EVAL_IMAGES_DIR"

for subset in ImageNet_1K HatefulMemes VOC2007 N24News SUN397; do
    if [[ ! -d "${TRAIN_IMAGES_DIR}/${subset}" ]]; then
        unzip -q -o "${DATASETS_ROOT}/${subset}.zip" -d "$TRAIN_IMAGES_DIR"
    fi
done

if ! compgen -G "${EVAL_IMAGES_DIR}/*" > /dev/null; then
    unzip -q -o "${DATASETS_ROOT}/images.zip" -d "$EVAL_IMAGES_DIR"
fi

# Run sequentially so a train/eval failure stops the pipeline.
bash scripts/cls/run_segd_cls.sh

EVAL_MODEL="${MODEL:-training/${EXP_NAME}/checkpoint-final}"
if [[ ! -f "${EVAL_MODEL}/config.json" ]]; then
    echo "ERROR: training checkpoint not found: ${EVAL_MODEL}" >&2
    exit 1
fi

MODEL="$EVAL_MODEL" bash scripts/cls/eval_segd_cls.sh

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
