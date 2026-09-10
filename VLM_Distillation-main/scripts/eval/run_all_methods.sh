#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_DIR}/outputs}"
command -v python >/dev/null 2>&1 || {
  echo "ERROR: python is not available. Activate the evaluation environment first." >&2; exit 2; }

ARGS=("$@")
# Backward compatible: run_all_methods.sh SUITE [PATTERN]
if [[ ${#ARGS[@]} -ge 1 && "${ARGS[0]}" != --* ]]; then
  LEGACY=(--suite "${ARGS[0]}")
  [[ ${#ARGS[@]} -lt 2 ]] || LEGACY+=(--pattern "${ARGS[1]}")
  ARGS=("${LEGACY[@]}")
fi
exec python "${SCRIPT_DIR}/run_all_methods.py" \
  --project-dir "${PROJECT_DIR}" --outputs-root "${OUTPUTS_ROOT}" \
  "${ARGS[@]}"
