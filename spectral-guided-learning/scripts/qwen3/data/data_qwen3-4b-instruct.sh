#!/usr/bin/env bash
# Phase 2: data prep for the Qwen3-4B-Instruct-2507 track.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f .venv/bin/activate ]] || ./scripts/setup.sh
  source .venv/bin/activate
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "data/qwen3-4b-instruct"

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/spectral-guided-learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/spectral-guided-learning}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-4B-Instruct-2507"
DATASET_NAME="${LOCAL_DATA_ROOT}/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_PATH="data/qwen3-4b-instruct/train-s1k-segmented.jsonl"
N_SAMPLES=1050
MAX_TOKENS=32768

OPTS=""
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --tokenizer ${MODEL_NAME}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --chat-template"
OPTS+=" --no-enable-thinking"

CMD="python ${BASE_PATH}/src/data_prep.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen3-4b-instruct-data.log
