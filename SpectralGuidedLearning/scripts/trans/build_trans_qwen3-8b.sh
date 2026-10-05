#!/usr/bin/env bash
# L_trans data: add \n\n-step fields to a masked Qwen3-8B dataset (vanilla by default).
#   bash scripts/trans/build_trans_qwen3-8b.sh            # train-vanilla  -> train-vanilla-trans.jsonl  (SFT + L_trans)
#   bash scripts/trans/build_trans_qwen3-8b.sh spectral   # train-spectral -> train-spectral-trans.jsonl (SGL + L_trans)
# Needs data/qwen3-8b/train-segmented.jsonl (data_qwen3-8b.sh) and the masked file (masks/iwc scripts).
set -euo pipefail

VARIANT="${1:-vanilla}"
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-8B"
DATA_DIR="data/qwen3-8b"
MIN_STEP_TOKENS="${MIN_STEP_TOKENS:-8}"

CMD="python -m sgl.transforms.build_transitions \
  --segmented ${DATA_DIR}/train-segmented.jsonl \
  --masked ${DATA_DIR}/train-${VARIANT}.jsonl \
  --output ${DATA_DIR}/train-${VARIANT}-trans.jsonl \
  --tokenizer ${MODEL_NAME} \
  --min-step-tokens ${MIN_STEP_TOKENS}"
echo "${CMD}"
${CMD} 2>&1 | tee "logs/qwen3-8b-trans-data-${VARIANT}.log"
