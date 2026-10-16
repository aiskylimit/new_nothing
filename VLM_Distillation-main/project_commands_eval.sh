#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
EVAL_ENV=/mnt/local/uvenvs/vlm-distill-eval/bin/activate
DEFAULT_BASE_MODEL=/mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/KamilaMila/FastVLM-0.5B

usage() {
  echo "Usage: $0 OUTPUT_DIR_OR_CHECKPOINT [BASE_MODEL] [RUN_EVAL_ARGS ...]" >&2
}

[[ $# -ge 1 ]] || { usage; exit 2; }

checkpoint_source="$1"
base_model="${2:-${BASE_MODEL:-${DEFAULT_BASE_MODEL}}}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

source "${EVAL_ENV}"
cd "${PROJECT_DIR}"

# Eval must see exactly one physical GPU. Inside the process, GPU 4 becomes
# logical cuda:0, which is what device_map=auto will use.
export CUDA_VISIBLE_DEVICES=4
export LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true

[[ -d "${checkpoint_source}" ]] || {
  echo "ERROR: checkpoint/output directory does not exist: ${checkpoint_source}" >&2
  exit 2
}
[[ -s "${base_model}/config.json" ]] || {
  echo "ERROR: invalid or incomplete FastVLM base model: ${base_model}" >&2
  exit 2
}

# Accept either an exact adapter checkpoint or a run output directory. For a
# run directory, select the valid checkpoint with the largest numeric step.
if [[ -s "${checkpoint_source}/adapter_config.json" ]]; then
  trained_checkpoint="${checkpoint_source}"
else
  trained_checkpoint=""
  latest_step=-1
  shopt -s nullglob
  for candidate in "${checkpoint_source}"/checkpoint-*; do
    [[ -d "${candidate}" && -s "${candidate}/adapter_config.json" ]] || continue
    checkpoint_name="${candidate##*/}"
    checkpoint_step="${checkpoint_name#checkpoint-}"
    [[ "${checkpoint_step}" =~ ^[0-9]+$ ]] || continue
    if (( checkpoint_step > latest_step )); then
      latest_step="${checkpoint_step}"
      trained_checkpoint="${candidate}"
    fi
  done
  shopt -u nullglob
fi

[[ -n "${trained_checkpoint}" ]] || {
  echo "ERROR: no valid adapter checkpoint found under: ${checkpoint_source}" >&2
  exit 2
}

printf '[%s] Evaluating checkpoint %s with physical GPU 4\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${trained_checkpoint}"

SUITE_CONFIG="${PROJECT_DIR}/configs/eval/requested_benchmarks.json" \
  bash "${PROJECT_DIR}/scripts/eval/prepare_eval_assets.sh" --offline

exec bash "${PROJECT_DIR}/scripts/eval/run_eval.sh" \
  --checkpoint "${trained_checkpoint}" \
  --base-model "${base_model}" \
  --suite requested_benchmarks \
  --mode all \
  --attention-backend eager \
  --max-new-tokens 128 \
  "$@"
