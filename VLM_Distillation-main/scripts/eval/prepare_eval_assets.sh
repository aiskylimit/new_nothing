#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${PROJECT_DIR}/VLMEvalKit}"
LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"
SUITE_CONFIG="${SUITE_CONFIG:-${PROJECT_DIR}/configs/eval/requested_benchmarks.json}"

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is not available. Activate the evaluation environment first." >&2
  exit 2
fi
if [[ ! -f "${VLMEVALKIT_DIR}/run.py" ]] || \
   { [[ ! -d "${VLMEVALKIT_DIR}/.git" ]] && [[ ! -f "${VLMEVALKIT_DIR}/.vendored-commit" ]]; }; then
  echo "ERROR: vendored VLMEvalKit source/metadata not found at ${VLMEVALKIT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SUITE_CONFIG}" ]]; then
  echo "ERROR: suite config not found: ${SUITE_CONFIG}" >&2
  exit 2
fi

exec python "${SCRIPT_DIR}/prepare_eval_assets.py" \
  --project-dir "${PROJECT_DIR}" \
  --vlmevalkit-dir "${VLMEVALKIT_DIR}" \
  --lmu-data "${LMUData}" \
  --suite-config "${SUITE_CONFIG}" \
  "$@"
