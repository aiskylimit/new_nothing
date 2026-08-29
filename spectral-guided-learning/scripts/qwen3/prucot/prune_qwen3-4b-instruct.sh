#!/usr/bin/env bash
# Phase 4 (Pru-CoT baseline): LLM-guided pruning for the Qwen3-4B-Instruct-2507 track (agent:
# Qwen2.5-3B-Instruct, scale-matched to the student per paper Section 4.4 -- an oversized agent prunes steps the student still needs).
set -euo pipefail

GPUS=(6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

DATA_PATH="${BASE_PATH}/data/qwen3-4b-instruct/train-s1k-segmented.jsonl"
WEIGHTS_PATH="${BASE_PATH}/data/qwen3-4b-instruct/prucot-weights.parquet"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
TOKENIZER="${LOCAL_MODELS_ROOT}/Qwen3-4B-Instruct-2507"
PRUNING_AGENT="${LOCAL_MODELS_ROOT}/Qwen2.5-3B-Instruct"
OUTPUT_PATH="${BASE_PATH}/data/qwen3-4b-instruct/train-prucot.jsonl"
CANDIDATE_THRESHOLD=0.5
MEDIAN_GATE_THRESHOLD=1.0
# The pruning agents are all Qwen2.5 (1.5B/3B/7B), whose context is 32768 -- vLLM aborts at
# startup if max_model_len exceeds that. 32768 is the hard ceiling here, not a tunable.
MAX_MODEL_LEN=32768
# vLLM tensor-parallel across every GPU in GPUS (must equal the GPU count, or vLLM aborts).
TENSOR_PARALLEL_SIZE=${#GPUS[@]}

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
${CMD} 2>&1 | tee "${BASE_PATH}/logs/qwen3-4b-instruct-prucot-prune.log"

echo ">>> STOP AND READ: check the step/token drop table above -- ~0% means prucot == vanilla."
