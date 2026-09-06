#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_DIR}/outputs}"
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: missing repository Python: ${PYTHON_BIN}" >&2; exit 2; }

ARGS=("$@")
# Backward compatible: run_all_methods.sh SUITE [PATTERN]
if [[ ${#ARGS[@]} -ge 1 && "${ARGS[0]}" != --* ]]; then
  LEGACY=(--suite "${ARGS[0]}")
  [[ ${#ARGS[@]} -lt 2 ]] || LEGACY+=(--pattern "${ARGS[1]}")
  ARGS=("${LEGACY[@]}")
fi
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all_methods.py" \
  --project-dir "${PROJECT_DIR}" --python-bin "${PYTHON_BIN}" --outputs-root "${OUTPUTS_ROOT}" \
  "${ARGS[@]}"
