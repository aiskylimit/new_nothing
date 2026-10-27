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
OPD_PROFILE="${OPD_PROFILE:-llama2-tinyllama}"
if [[ "$ACTION" == download ]]; then
  case "$OPD_PROFILE" in
    llama2-tinyllama) MANIFEST=download.txt ;;
    math-small) MANIFEST=download_math.txt ;;
    tiny) MANIFEST=download_tiny.txt ;;
    *) echo "Unknown OPD_PROFILE: $OPD_PROFILE" >&2; exit 1 ;;
  esac
  python -m opd.local_assets download --manifest "$SCRIPT_DIR/$MANIFEST" --project-root "$OPD_PROJECT_ROOT"
  exit 0
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ "$ACTION" == all || "$ACTION" == prepare ]]; then
  python -m opd.local_assets prepare --project-root "$OPD_PROJECT_ROOT" --profile "$OPD_PROFILE"
fi
if [[ "$ACTION" == prepare ]]; then exit 0; fi
read -r -a OPD_DOMAINS <<< "${DOMAINS:-dolly-dskd}"
read -r -a OPD_SEEDS <<< "${SEEDS:-10}"
read -r -a OPD_BI_SEEDS <<< "${BI_SEEDS:-42 43 44}"
read -r -a OPD_METHODS <<< "${METHODS:-opd opd_impact}"
OPD_EXTRA=()
if [[ -n "${TEACHER_ADAPTER:-}" ]]; then OPD_EXTRA+=(--teacher-adapter "$TEACHER_ADAPTER"); fi
if [[ "${MATH_EVAL:-0}" == 1 ]]; then OPD_EXTRA+=(--math-eval); fi
python -m opd.b200 "$ACTION" --project-root "$OPD_PROJECT_ROOT" \
  --run-name "${RUN_NAME:-llama2-tinyllama-opd}" --profile "$OPD_PROFILE" \
  --domains "${OPD_DOMAINS[@]}" --seeds "${OPD_SEEDS[@]}" \
  --bi-seeds "${OPD_BI_SEEDS[@]}" --steps "${STEPS:-0}" \
  --grad-accum "${GRAD_ACCUM:-0}" --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}" \
  --max-minutes "${MAX_MINUTES:-0}" --teacher-steps "${TEACHER_STEPS:-0}" \
  --methods "${OPD_METHODS[@]}" "${OPD_EXTRA[@]}"
