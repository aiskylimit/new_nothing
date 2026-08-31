#!/usr/bin/env bash
# Phase 2: data prep for the Qwen2.5-7B-Instruct track.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "data/qwen25-7b"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothing}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen2.5-7B-Instruct"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B}"
OUTPUT_PATH="data/qwen25-7b/train-segmented.jsonl"
N_SAMPLES="${N_SAMPLES:-}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

OPTS=""
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --tokenizer ${MODEL_NAME}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --chat-template"
OPTS+=" --no-enable-thinking"
if [[ -n "${N_SAMPLES}" ]]; then
  OPTS+=" --n-samples ${N_SAMPLES}"
fi

CMD="python ${BASE_PATH}/src/data_prep.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen25-7b-data.log
