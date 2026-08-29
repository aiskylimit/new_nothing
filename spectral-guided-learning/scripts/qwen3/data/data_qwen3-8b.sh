#!/usr/bin/env bash
# Phase 2: data prep for the Qwen3-8B track.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "data/qwen3-8b"

MODEL_NAME="Qwen/Qwen3-8B"
DATASET_NAME="VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_PATH="data/qwen3-8b/train-s1k-segmented.jsonl"
N_SAMPLES=1050
MAX_TOKENS=32768

OPTS=""
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --tokenizer ${MODEL_NAME}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --chat-template"
OPTS+=" --enable-thinking"

CMD="python ${BASE_PATH}/src/data_prep.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen3-8b-data.log
