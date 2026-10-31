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
OPD_FAST=0
if [[ "$ACTION" == fast ]]; then ACTION=all; OPD_FAST=1; fi
# User-facing train includes the post-training evaluation by default.
if [[ "$ACTION" == train && "${AUTO_EVAL:-1}" == 1 ]]; then ACTION=train-eval; fi
OPD_PROFILE="${OPD_PROFILE:-llama2-tinyllama}"
OPD_RUN_NAME="${RUN_NAME:-llama2-tinyllama-opd-8k}"
if [[ "$ACTION" == setup-eval ]]; then
  uv pip install --python "$OPD_VENV/bin/python" -r "$SCRIPT_DIR/requirements-opd-dolly-eval.txt"
  exit 0
fi
if [[ "$ACTION" == download-eval ]]; then
  python -m opd.benchmarks download --project-root "$OPD_PROJECT_ROOT"
  exit 0
fi
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
if [[ -n "${ONLY_METHOD:-}" ]]; then OPD_EXTRA+=(--only-method "$ONLY_METHOD"); fi
if [[ -n "${STOP_AT:-}" ]]; then OPD_EXTRA+=(--stop-at "$STOP_AT"); fi
if [[ "$OPD_FAST" == 1 || "${FAST:-0}" == 1 ]]; then
  OPD_EXTRA+=(--fast --micro-batch-size "${MICRO_BATCH_SIZE:-16}")
  if [[ "${GRADIENT_CHECKPOINTING:-0}" == 1 ]]; then OPD_EXTRA+=(--gradient-checkpointing); fi
fi
if [[ -n "${TEACHER_ADAPTER:-}" ]]; then
  OPD_EXTRA+=(--teacher-adapter "$TEACHER_ADAPTER")
elif [[ "$OPD_PROFILE" == llama2-tinyllama && "$OPD_RUN_NAME" == llama2-tinyllama-opd-8k ]]; then
  OPD_TEACHER_SOURCE="${TEACHER_FROM_RUN-llama2-tinyllama-opd}"
  if [[ -n "$OPD_TEACHER_SOURCE" ]]; then OPD_EXTRA+=(--teacher-from-run "$OPD_TEACHER_SOURCE"); fi
fi
OPD_EXTRA+=(--train-samples "${TRAIN_SAMPLES:-8000}" --train-subset-seed "${TRAIN_SUBSET_SEED:-42}")
if [[ "${MATH_EVAL:-0}" == 1 ]]; then OPD_EXTRA+=(--math-eval); fi
if [[ "${DOLLY_EVAL:-1}" == 0 ]]; then OPD_EXTRA+=(--no-dolly-eval); fi
read -r -a OPD_EVAL_SEEDS <<< "${DOLLY_EVAL_SEEDS:-10 20 30 40 50}"
OPD_EXTRA+=(--dolly-eval-batch-size "${EVAL_BATCH_SIZE:-16}" --dolly-eval-seeds "${OPD_EVAL_SEEDS[@]}")
read -r -a OPD_BENCHMARKS <<< "${EVAL_BENCHMARKS:-dolly self-inst vicuna sinst}"
OPD_EXTRA+=(--eval-benchmarks "${OPD_BENCHMARKS[@]}")
python -m opd.b200 "$ACTION" --project-root "$OPD_PROJECT_ROOT" \
  --run-name "$OPD_RUN_NAME" --profile "$OPD_PROFILE" \
  --domains "${OPD_DOMAINS[@]}" --seeds "${OPD_SEEDS[@]}" \
  --bi-seeds "${OPD_BI_SEEDS[@]}" --steps "${STEPS:-0}" \
  --grad-accum "${GRAD_ACCUM:-0}" --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}" \
  --max-minutes "${MAX_MINUTES:-0}" --teacher-steps "${TEACHER_STEPS:-0}" \
  --methods "${OPD_METHODS[@]}" "${OPD_EXTRA[@]}"
