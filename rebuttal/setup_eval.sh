#!/usr/bin/env bash
# Shared eval dependencies only; never reinstall a native training environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_ROOT="${REBUTTAL_ENV_ROOT:-/mnt/local/uvenvs}"
EVAL_ENV="${REBUTTAL_EVAL_VENV:-$ENV_ROOT/rebuttal_eval}"
EVAL_PY="$EVAL_ENV/bin/python"
if [[ "${REBUTTAL_EVAL_PYTHON:-$EVAL_PY}" != "$EVAL_PY" ]]; then
  echo 'Set REBUTTAL_EVAL_VENV to the environment to install; REBUTTAL_EVAL_PYTHON must match its bin/python.' >&2
  exit 2
fi
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv' >&2; exit 1; }
[[ -x "$EVAL_PY" ]] || uv venv --python 3.11 "$EVAL_ENV"
uv pip install --python "$EVAL_PY" -r rebuttal_eval.txt
uv pip check --python "$EVAL_PY"
echo "Shared evaluator installed: $EVAL_PY (run check-eval on the GPU node before training)"
