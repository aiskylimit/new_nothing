#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPD_PROJECT_ROOT="${OPD_PROJECT_ROOT:-$SCRIPT_DIR}"
OPD_VENV="${OPD_VENV:-/mnt/local/uvenvs/impact-opd-b200}"
source "$OPD_VENV/bin/activate"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$OPD_PROJECT_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ACTION="${1:-all}"
if [[ "$ACTION" == download ]]; then
  python -m opd.local_assets download --manifest "$SCRIPT_DIR/download.txt" --project-root "$OPD_PROJECT_ROOT"
  if [[ "${OPD_PROFILE:-math-small}" == tiny ]]; then
    python -m opd.local_assets download --manifest "$SCRIPT_DIR/download_tiny.txt" --project-root "$OPD_PROJECT_ROOT"
  fi
  exit 0
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ "$ACTION" == all || "$ACTION" == prepare ]]; then
  python -m opd.local_assets prepare --project-root "$OPD_PROJECT_ROOT"
fi
if [[ "$ACTION" == prepare ]]; then exit 0; fi
read -r -a OPD_DOMAINS <<< "${DOMAINS:-gsm8k mixed dolly}"
read -r -a OPD_SEEDS <<< "${SEEDS:-42}"
read -r -a OPD_BI_SEEDS <<< "${BI_SEEDS:-42 43 44}"
python -m opd.b200 "$ACTION" --project-root "$OPD_PROJECT_ROOT" \
  --run-name "${RUN_NAME:-opd-b200}" --profile "${OPD_PROFILE:-math-small}" \
  --domains "${OPD_DOMAINS[@]}" --seeds "${OPD_SEEDS[@]}" \
  --bi-seeds "${OPD_BI_SEEDS[@]}" --steps "${STEPS:-256}" \
  --grad-accum "${GRAD_ACCUM:-8}" --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}" \
  --max-minutes "${MAX_MINUTES:-0}"
