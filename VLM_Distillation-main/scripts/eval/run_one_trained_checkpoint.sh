#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"

# Either edit these two defaults, set the matching environment variables, or
# pass both paths as positional arguments (positional arguments take priority).
DEFAULT_TRAINED_CHECKPOINT="/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/CHANGE_ME/checkpoint-CHANGE_ME"
DEFAULT_BASE_MODEL="/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/CHANGE_ME"
TRAINED_CHECKPOINT="${1:-${TRAINED_CHECKPOINT:-${DEFAULT_TRAINED_CHECKPOINT}}}"
BASE_MODEL="${2:-${BASE_MODEL:-${DEFAULT_BASE_MODEL}}}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  set --
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: missing repository Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -s "${TRAINED_CHECKPOINT}/adapter_config.json" ]] || {
  echo "ERROR: not an adapter-only LoRA checkpoint: ${TRAINED_CHECKPOINT}" >&2
  exit 2
}
[[ -s "${BASE_MODEL}/config.json" ]] || {
  echo "ERROR: invalid or incomplete local base model: ${BASE_MODEL}" >&2
  exit 2
}

export LMUData HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true
SUITE_CONFIG="${PROJECT_DIR}/configs/eval/requested_benchmarks.json" \
  bash "${SCRIPT_DIR}/prepare_eval_assets.sh" --offline

exec bash "${SCRIPT_DIR}/run_eval.sh" \
  --checkpoint "${TRAINED_CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --suite requested_benchmarks \
  --mode all \
  --attention-backend sdpa \
  --max-new-tokens 128 \
  "$@"
