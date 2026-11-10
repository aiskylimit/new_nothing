#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SCRIPT="$PWD/rebuttal/project_commands.sh"
ENV_ROOT="${REBUTTAL_ENV_ROOT:-/mnt/local/uvenvs}"
export REBUTTAL_VENV="${REBUTTAL_VENV:-$ENV_ROOT/rebuttal}"
export REBUTTAL_PYTHON_LONGCOT="${REBUTTAL_PYTHON_LONGCOT:-$ENV_ROOT/sft/bin/python}"
export REBUTTAL_PYTHON_PALIGN="${REBUTTAL_PYTHON_PALIGN:-$ENV_ROOT/palign/bin/python}"
export REBUTTAL_PYTHON_SGL="${REBUTTAL_PYTHON_SGL:-$ENV_ROOT/sgl/bin/python}"
export REBUTTAL_PYTHON_SSFT="${REBUTTAL_PYTHON_SSFT:-$ENV_ROOT/SegmentSelectiveSFT/bin/python}"
export REBUTTAL_PYTHON_CORED="${REBUTTAL_PYTHON_CORED:-$ENV_ROOT/cored/bin/python}"
EVAL_ENV="${REBUTTAL_EVAL_VENV:-$ENV_ROOT/rebuttal_eval}"
export REBUTTAL_EVAL_PYTHON="${REBUTTAL_EVAL_PYTHON:-$EVAL_ENV/bin/python}"
ACTION="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
# Setup must work before the environment exists. End-to-end delegates activation
# to each child stage, after optional online setup has created the environment.
if [[ "$ACTION" != setup && "$ACTION" != endtoend && -f "$REBUTTAL_VENV/bin/activate" ]]; then
  source "$REBUTTAL_VENV/bin/activate"
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python}"
ASSETS="${REBUTTAL_ASSETS:-$PWD/rebuttal/assets}"
DATA="${REBUTTAL_DATA:-$PWD/rebuttal/prepared}"
RUNS="${REBUTTAL_RUNS:-$PWD/rebuttal/runs}"
REPORTS="${REBUTTAL_REPORTS:-$PWD/rebuttal/reports}"
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
      exec "$PY" -m rebuttal.eval_preflight --assets "$ASSETS" --runs "$RUNS" --gpus "${GPUS:-3,4,5}" "$@"
    ;;
  download)
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
      exec "$PY" -m rebuttal.provision --assets "$ASSETS" "$@"
    ;;
  endtoend)
    MODE=--offline
    if [[ $# -gt 0 ]]; then MODE="$1"; shift; fi
    if [[ $# -ne 0 || ( "$MODE" != --online && "$MODE" != --offline ) ]]; then
      echo 'Usage: bash rebuttal/project_commands.sh endtoend [--online|--offline]' >&2
      exit 2
    fi
    LOG_DIR="${REBUTTAL_LAUNCHER_LOGS:-$PWD/rebuttal/launcher-logs}"
    mkdir -p "$LOG_DIR"
    run_stage() {
      echo "[B200 end-to-end] $1" | tee -a "$LOG_DIR/endtoend.log"
      bash "$SCRIPT" "$@" 2>&1 | tee -a "$LOG_DIR/endtoend.log"
    }
    if [[ "$MODE" == --online ]]; then
      run_stage setup
      run_stage download
    fi
    for stage in prepare preflight all report; do
      run_stage "$stage"
    done
    ;;
  prepare) exec "$PY" -m rebuttal.data --assets "$ASSETS" --out "$DATA" "$@" ;;
  preflight) exec "$PY" -m rebuttal.preflight --assets "$ASSETS" --data "$DATA" --runs "$RUNS" --environments --gpus "${GPUS:-3,4,5}" "$@" ;;
  configs|train|eval|all|smoke) exec "$PY" -m rebuttal.runner "$ACTION" --assets "$ASSETS" --data "$DATA" --runs "$RUNS" --gpus "${GPUS:-3,4,5}" "$@" ;;
  report)
    "$PY" -m rebuttal.statistics --runs "$RUNS" --output "$REPORTS" "$@"
    exec "$PY" -m rebuttal.cost --runs "$RUNS" --output "$REPORTS"
    ;;
  *) echo "Usage: $0 {setup|setup-eval|check-eval|download|endtoend [--online|--offline]|prepare|preflight|smoke|train|eval|all|report|configs}" >&2; exit 2 ;;
esac
