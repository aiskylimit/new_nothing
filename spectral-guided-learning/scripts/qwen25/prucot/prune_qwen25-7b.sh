#!/usr/bin/env bash
# Phase 4 (Pru-CoT baseline): LLM-guided pruning for the Qwen2.5-7B-Instruct track.
set -euo pipefail

GPUS=(2 3)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${BASE_PATH}/.venv/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${BASE_PATH}/.venv/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

DATA_PATH="${BASE_PATH}/data/qwen25-7b/train-s1k-segmented.jsonl"
WEIGHTS_PATH="${BASE_PATH}/data/qwen25-7b/prucot-weights.parquet"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
PRUNING_AGENT="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH="${BASE_PATH}/data/qwen25-7b/train-prucot.jsonl"
CANDIDATE_THRESHOLD=0.5
MEDIAN_GATE_THRESHOLD=1.0
MAX_MODEL_LEN=40960
TENSOR_PARALLEL_SIZE=2

OPTS=""
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --weights-path ${WEIGHTS_PATH}"
OPTS+=" --tokenizer ${TOKENIZER}"
OPTS+=" --pruning-agent ${PRUNING_AGENT}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --candidate-threshold ${CANDIDATE_THRESHOLD}"
OPTS+=" --median-gate-threshold ${MEDIAN_GATE_THRESHOLD}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"

CMD="python ${BASE_PATH}/src/prucot_prune.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/qwen25-7b-prucot-prune.log"

echo ">>> STOP AND READ: check the step/token drop table above -- ~0% means prucot == vanilla."
