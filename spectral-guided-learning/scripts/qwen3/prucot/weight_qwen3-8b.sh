#!/usr/bin/env bash
# Phase 3 (Pru-CoT baseline): step-importance global optimization for the Qwen3-8B track.
set -euo pipefail

GPUS=(2 3)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f .venv/bin/activate ]] || ./scripts/setup.sh
  source .venv/bin/activate
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

MODEL_NAME="Qwen/Qwen3-8B"
DATA_PATH="data/qwen3-8b/train-s1k-segmented.jsonl"
OUTPUT_DIR="data/qwen3-8b/prucot"
WEIGHTS_PATH="data/qwen3-8b/prucot-weights.parquet"
EPOCHS=3
LR=10
FILLER_TOKEN="."

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --weights-path ${WEIGHTS_PATH}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --lr ${LR}"
OPTS+=" --filler-token ${FILLER_TOKEN}"

CMD="python ${BASE_PATH}/src/prucot_weight.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen3-8b-prucot-weight.log
