#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -n "${REBUTTAL_VENV:-}" ]]; then
  source "$REBUTTAL_VENV/bin/activate"
elif [[ -f /mnt/local/uvenvs/rebuttal/bin/activate ]]; then
  source /mnt/local/uvenvs/rebuttal/bin/activate
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python}"
ASSETS="${REBUTTAL_ASSETS:-$PWD/rebuttal/assets}"
DATA="${REBUTTAL_DATA:-$PWD/rebuttal/prepared}"
RUNS="${REBUTTAL_RUNS:-$PWD/rebuttal/runs}"
REPORTS="${REBUTTAL_REPORTS:-$PWD/rebuttal/reports}"
ACTION="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  prepare) exec "$PY" -m rebuttal.data --assets "$ASSETS" --out "$DATA" "$@" ;;
  preflight) exec "$PY" -m rebuttal.preflight --assets "$ASSETS" --data "$DATA" --runs "$RUNS" --environments --gpus "${GPUS:-3,4,5}" "$@" ;;
  configs|train|eval|all|smoke) exec "$PY" -m rebuttal.runner "$ACTION" --assets "$ASSETS" --data "$DATA" --runs "$RUNS" --gpus "${GPUS:-3,4,5}" "$@" ;;
  report)
    "$PY" -m rebuttal.statistics --runs "$RUNS" --output "$REPORTS" "$@"
    exec "$PY" -m rebuttal.cost --runs "$RUNS" --output "$REPORTS"
    ;;
  *) echo "Usage: $0 {prepare|preflight|smoke|train|eval|all|report|configs}" >&2; exit 2 ;;
esac
