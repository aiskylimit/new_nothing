#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${PROJECT_DIR}/VLMEvalKit}"
LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: missing repository Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${VLMEVALKIT_DIR}/run.py" ]] && \
  { [[ -d "${VLMEVALKIT_DIR}/.git" ]] || [[ -f "${VLMEVALKIT_DIR}/.vendored-commit" ]]; } || {
  echo "ERROR: vendored VLMEvalKit source/metadata not found at ${VLMEVALKIT_DIR}" >&2; exit 2; }

ARGS=("$@")
# Retain the old: run_eval.sh SUITE CHECKPOINT [MODEL_NAME]
if [[ ${#ARGS[@]} -ge 2 && "${ARGS[0]}" != --* ]]; then
  LEGACY=(--suite "${ARGS[0]}" --checkpoint "${ARGS[1]}")
  [[ ${#ARGS[@]} -lt 3 ]] || LEGACY+=(--model-name "${ARGS[2]}")
  ARGS=("${LEGACY[@]}")
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_eval.py" \
  --project-dir "${PROJECT_DIR}" --python-bin "${PYTHON_BIN}" \
  --vlmevalkit-dir "${VLMEVALKIT_DIR}" --lmu-data "${LMUData}" \
  "${ARGS[@]}"
