#!/usr/bin/env bash
# Phase 4: build masks for the Qwen2.5-7B-Instruct track (no --vanilla: compares against
# P-ALIGN's published vanilla-SFT numbers, not a local run).
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
ENERGY_THRESHOLD_P=0.95

OPTS=""
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --strengths ${STRENGTHS_PATH}"
OPTS+=" --energy-threshold-p ${ENERGY_THRESHOLD_P}"

CMD="python ${BASE_PATH}/src/build_masks.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen25-7b-masks.log

echo ">>> STOP AND READ: check the step/token drop table above -- ~0% means spectral == vanilla."
