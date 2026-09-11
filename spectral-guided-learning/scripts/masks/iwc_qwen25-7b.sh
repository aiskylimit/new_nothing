#!/usr/bin/env bash
# Build both IWC datasets after spectral capture for Qwen2.5-7B-Instruct.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

DATA_PATH="data/qwen25-7b/train-segmented.jsonl"
STRENGTHS_PATH="data/qwen25-7b/spectral-strengths.parquet"
ENERGY_THRESHOLD_P="${IWC_ENERGY_THRESHOLD_P:-0.95}"
TEMPERATURE="${IWC_TEMPERATURE:-1.0}"
INTERPOLATION="${IWC_INTERPOLATION:-1.0}"
CLIP="${IWC_CLIP:-2.0}"

CMD="python ${BASE_PATH}/src/build_iwc_datasets.py --data-path ${DATA_PATH} --strengths ${STRENGTHS_PATH} --energy-threshold-p ${ENERGY_THRESHOLD_P} --temperature ${TEMPERATURE} --interpolation ${INTERPOLATION} --clip ${CLIP}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen25-7b-iwc-masks.log
