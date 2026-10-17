#!/usr/bin/env bash
# Phase 2+4 for the answer-only SFT arm: s1K-1.1 with the long CoT dropped, only the ground-truth
# `solution` field (no DeepSeek/Gemini output at all) as the target. No capture/spectral phase is needed, so the all-ones
# mask is emitted right here (build_masks.py --vanilla-only).
#   bash scripts/data/data_answer.sh qwen25-7b
#   bash scripts/data/data_answer.sh qwen3-8b
#   bash scripts/data/data_answer.sh r1-qwen-1.5b
set -euo pipefail

TRACK="${1:?track is required: qwen25-7b, qwen3-8b or r1-qwen-1.5b}"
case "${TRACK}" in
  qwen25-7b) MODEL_DIR="Qwen2.5-7B-Instruct" ;;
  qwen3-8b) MODEL_DIR="Qwen3-8B" ;;
  r1-qwen-1.5b) MODEL_DIR="DeepSeek-R1-Distill-Qwen-1.5B" ;;
  *) echo "unknown track: ${TRACK}" >&2; exit 2 ;;
esac

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
DATA_DIR="data/${TRACK}-answer"
mkdir -p logs "${DATA_DIR}"

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/${MODEL_DIR}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
OUTPUT_PATH="${DATA_DIR}/train-segmented.jsonl"
N_SAMPLES="${N_SAMPLES:-}"
# Same cap as the long-CoT track so both arms draw from the same shuffled stream; answer-only
# samples are far shorter, so effectively nothing is rejected here.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# OFF for every track, including R1-Distill: there is no reasoning to supervise here, so the
# thinking block is closed in the PROMPT (close_open_thinking) and the target is the bare
# solution -- exactly the prompt the eval scripts render (thinking OFF by default). Leaving it
# on would only teach the model to emit an empty "</think>" before the answer and would make the
# train prompt differ from the eval one. Set ENABLE_THINKING=true to get that variant instead.
ENABLE_THINKING="${ENABLE_THINKING:-false}"

OPTS=""
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --tokenizer ${MODEL_NAME}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --chat-template"
[[ "${ENABLE_THINKING}" == true ]] && OPTS+=" --enable-thinking" || OPTS+=" --no-enable-thinking"
OPTS+=" --response-mode answer"
if [[ -n "${N_SAMPLES}" ]]; then
  OPTS+=" --n-samples ${N_SAMPLES}"
fi

CMD="python -m sgl.data.prepare ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "logs/${TRACK}-answer-data.log"

CMD="python -m sgl.selection.build_masks --data-path ${OUTPUT_PATH} --vanilla-only"
echo "${CMD}"
${CMD} 2>&1 | tee "logs/${TRACK}-answer-masks.log"
