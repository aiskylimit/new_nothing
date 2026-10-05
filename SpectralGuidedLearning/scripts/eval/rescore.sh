#!/usr/bin/env bash
# Re-score an existing eval run from its saved generations (no GPU, no vLLM):
#   ./scripts/eval/rescore.sh vanilla-unsloth-qwen25-7b
# Reads results/<tag>/raw/*.jsonl, rewrites results/<tag>/summary.json, prints the metric table.
# RESULTS_DIR=<dir> (default results/) picks the tree, e.g. RESULTS_DIR=results-32k after reeval_all.sh.
set -euo pipefail
[[ -n "${1:-}" ]] || { echo "usage: $0 <tag>  (a dir under results/)" >&2; exit 1; }
TAG="$1"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ "${VIRTUAL_ENV:-}" != "${PROJECT_ENV}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: eval env not found at ${PROJECT_ENV}; build it from spectral_guided_learning.txt (repo root) or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

RUN_DIR="${RESULTS_DIR:-${BASE_PATH}/results}/${TAG}"
[[ -d "${RUN_DIR}/raw" ]] || { echo "ERROR: no generations at ${RUN_DIR}/raw -- run the eval script first" >&2; exit 1; }

python -m sgl.eval.evaluate --rescore "${RUN_DIR}" 2>&1 | tee "${BASE_PATH}/logs/rescore-${TAG}.log"
