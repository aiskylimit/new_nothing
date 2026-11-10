#!/usr/bin/env bash
# Isolated end-to-end development smoke on 2 GPUs. Never produces main results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_ROOT="${REBUTTAL_ENV_ROOT:-/mnt/local/uvenvs}"
export REBUTTAL_VENV="${REBUTTAL_VENV:-$ENV_ROOT/rebuttal}"
export REBUTTAL_PYTHON_LONGCOT="${REBUTTAL_PYTHON_LONGCOT:-$ENV_ROOT/sft/bin/python}"
export REBUTTAL_PYTHON_PALIGN="${REBUTTAL_PYTHON_PALIGN:-$ENV_ROOT/palign/bin/python}"
export REBUTTAL_PYTHON_SGL="${REBUTTAL_PYTHON_SGL:-$ENV_ROOT/sgl/bin/python}"
export REBUTTAL_PYTHON_SSFT="${REBUTTAL_PYTHON_SSFT:-$ENV_ROOT/SegmentSelectiveSFT/bin/python}"
export REBUTTAL_PYTHON_CORED="${REBUTTAL_PYTHON_CORED:-$ENV_ROOT/cored/bin/python}"
EVAL_ENV="${REBUTTAL_EVAL_VENV:-$ENV_ROOT/rebuttal_eval}"
export REBUTTAL_EVAL_PYTHON="${REBUTTAL_EVAL_PYTHON:-$EVAL_ENV/bin/python}"
PY="${PY:-$REBUTTAL_VENV/bin/python}"
ASSETS="${REBUTTAL_ASSETS:-$PWD/rebuttal/assets}"
RUNS="${REBUTTAL_2GPU_RUNS:-$PWD/rebuttal/smoke-2x3090}"
ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  setup)
    command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv' >&2; exit 1; }
    mkdir -p "$ENV_ROOT"
    for name in rebuttal palign sft sgl SegmentSelectiveSFT cored; do
      [[ -x "$ENV_ROOT/$name/bin/python" ]] || uv venv --python 3.11 "$ENV_ROOT/$name"
      uv pip install --python "$ENV_ROOT/$name/bin/python" -r "$name.txt"
      uv pip check --python "$ENV_ROOT/$name/bin/python"
    done
    bash rebuttal/setup_eval.sh
    ;;
  setup-eval) exec bash rebuttal/setup_eval.sh "$@" ;;
  check-eval)
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      exec "$PY" -m rebuttal.eval_preflight --assets "$ASSETS" --runs "$RUNS" --gpus "${GPUS:-0,1}" "$@"
    ;;
  eval-existing)
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
      exec "$PY" -m rebuttal.eval_existing --gpus "${GPUS:-0,1}" "$@"
    ;;
  download)
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
      "$PY" -m rebuttal.provision --assets "$ASSETS" "$@"
    ;;
  all)
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
    export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
    mkdir -p "$(dirname "$RUNS")"
    "$PY" -m rebuttal.smoke_2gpu --assets "$ASSETS" --runs "$RUNS" --gpus "${GPUS:-0,1}" "$@" \
      2>&1 | tee -a "${RUNS}.launcher.log"
    ;;
  *) echo 'Usage: bash rebuttal/smoke_2x3090.sh {setup|setup-eval|check-eval|eval-existing|download|all} [options]' >&2; exit 2 ;;
esac
