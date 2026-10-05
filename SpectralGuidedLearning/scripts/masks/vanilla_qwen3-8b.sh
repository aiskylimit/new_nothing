#!/usr/bin/env bash
# Phase 4 (vanilla only): all-ones mask for the Qwen3-8B track -> data/qwen3-8b/train-vanilla.jsonl.
# No spectral-strengths.parquet needed, so it runs right after data_qwen3-8b.sh without the GPU
# capture phase; use masks_qwen3-8b.sh when the spectral arm's train-spectral.jsonl is needed too.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

CMD="python -m sgl.selection.build_masks --data-path data/qwen3-8b/train-segmented.jsonl --vanilla-only"
echo "${CMD}"
${CMD} 2>&1 | tee logs/qwen3-8b-masks-vanilla.log
