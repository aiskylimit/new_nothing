#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BASE_MODEL_EVAL_ROOT="${BASE_MODEL_EVAL_ROOT:-${PROJECT_DIR}/outputs/eval/base_models}"

exec python "${SCRIPT_DIR}/collect_eval_summaries.py" \
  --root "${BASE_MODEL_EVAL_ROOT}" \
  --csv "${BASE_MODEL_EVAL_ROOT}/combined_summary.csv" \
  --json "${BASE_MODEL_EVAL_ROOT}/combined_summary.json" \
  "$@"
