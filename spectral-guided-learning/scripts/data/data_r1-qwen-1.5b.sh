#!/usr/bin/env bash
# Phase 2: data prep (LIMO) -- DeepSeek-R1-Distill-Qwen-1.5B track.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "data/r1-qwen-1.5b"

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothing}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/LIMO}"
OUTPUT_PATH="data/r1-qwen-1.5b/train-segmented.jsonl"
N_SAMPLES="${N_SAMPLES:-}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

OPTS=""
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --tokenizer ${MODEL_NAME}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --chat-template"
OPTS+=" --enable-thinking"
if [[ -n "${N_SAMPLES}" ]]; then
  OPTS+=" --n-samples ${N_SAMPLES}"
fi

CMD="python ${BASE_PATH}/src/data_prep.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/r1-qwen-1.5b-data.log
