#!/usr/bin/env bash
# Phase 2: data prep for the Qwen2.5-7B-Instruct track.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f .venv/bin/activate ]] || ./scripts/setup.sh
  source .venv/bin/activate
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "data/qwen25-7b"

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME="VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_PATH="data/qwen25-7b/train-s1k-segmented.jsonl"
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
${CMD} 2>&1 | tee logs/qwen25-7b-data.log
