#!/usr/bin/env bash
# Build both IWC datasets after spectral capture for the DeepSeek-R1-Distill-Qwen-1.5B track.
set -euo pipefail
# Data/checkpoint namespace: r1-qwen-1.5b = s1K-1.1 track; e.g. r1-qwen-1.5b-palign = P-ALIGN data.
TRACK="${TRACK:-r1-qwen-1.5b}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE_PATH}/.." && pwd)/iwc}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

DATA_PATH="data/${TRACK}/train-segmented.jsonl"
STRENGTHS_PATH="data/${TRACK}/spectral-strengths.parquet"
ENERGY_THRESHOLD_P="${IWC_ENERGY_THRESHOLD_P:-0.95}"
TEMPERATURE="${IWC_TEMPERATURE:-1.0}"
INTERPOLATION="${IWC_INTERPOLATION:-1.0}"
CLIP="${IWC_CLIP:-2.0}"

CMD="python -m sgl.allocation.build --data-path ${DATA_PATH} --strengths ${STRENGTHS_PATH} --energy-threshold-p ${ENERGY_THRESHOLD_P} --temperature ${TEMPERATURE} --interpolation ${INTERPOLATION} --clip ${CLIP}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/${TRACK}-iwc-masks.log
