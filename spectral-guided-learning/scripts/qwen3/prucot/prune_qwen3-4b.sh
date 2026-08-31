#!/usr/bin/env bash
# Pru-CoT LLM-guided pruning, Qwen3-4B track (agent: Qwen2.5-3B-Instruct, scale-matched).
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

DATA_PATH="${BASE_PATH}/data/qwen3-4b/train-segmented.jsonl"
WEIGHTS_PATH="${BASE_PATH}/data/qwen3-4b/prucot-weights.parquet"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
TOKENIZER="${LOCAL_MODELS_ROOT}/Qwen3-4B"
PRUNING_AGENT="${LOCAL_MODELS_ROOT}/Qwen2.5-3B-Instruct"
OUTPUT_PATH="${BASE_PATH}/data/qwen3-4b/train-prucot.jsonl"
# Paper-faithful; no <8k-token sample filter -- we keep the full dataset.
CANDIDATE_THRESHOLD=0.5
MEDIAN_GATE_THRESHOLD=1.0
MAX_TOKENS=8192
MAX_MODEL_LEN=32768   # Qwen2.5 agent context ceiling; raised from paper's 16384 to fit long samples.
TENSOR_PARALLEL_SIZE=${#GPUS[@]}
# Empty = keep every sample. Must match weight_qwen3-4b.sh's PRUCOT_MAX_LENGTH.
MAX_LENGTH="${PRUCOT_MAX_LENGTH:-}"

OPTS=""
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --weights-path ${WEIGHTS_PATH}"
OPTS+=" --tokenizer ${TOKENIZER}"
OPTS+=" --pruning-agent ${PRUNING_AGENT}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --candidate-threshold ${CANDIDATE_THRESHOLD}"
OPTS+=" --median-gate-threshold ${MEDIAN_GATE_THRESHOLD}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
[[ -n "${MAX_LENGTH}" ]] && OPTS+=" --max-length ${MAX_LENGTH}"

CMD="python -u ${BASE_PATH}/src/prucot_prune.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/qwen3-4b-prucot-prune.log"

echo ">>> STOP AND READ: check the step/token drop table above -- ~0% means prucot == vanilla."
