#!/usr/bin/env bash
# Phase 4: build masks for the Qwen3-8B track (no --vanilla: compares against P-ALIGN's
# published vanilla-SFT numbers, not a local run).
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f .venv/bin/activate ]] || ./scripts/setup.sh
  source .venv/bin/activate
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

DATA_PATH="data/qwen3-8b/train-s1k-segmented.jsonl"
STRENGTHS_PATH="data/qwen3-8b/spectral-strengths.parquet"
ENERGY_THRESHOLD_P=0.95

OPTS=""
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --strengths ${STRENGTHS_PATH}"
OPTS+=" --energy-threshold-p ${ENERGY_THRESHOLD_P}"

CMD="python ${BASE_PATH}/src/build_masks.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen3-8b-masks.log

echo ">>> STOP AND READ: check the step/token drop table above -- ~0% means spectral == vanilla."
