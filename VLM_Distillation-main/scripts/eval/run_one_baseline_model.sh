#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"

# Edit this path, set BASE_MODEL, or pass the model directory as argument 1.
DEFAULT_BASE_MODEL="/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-3B-Instruct"
BASE_MODEL="${1:-${BASE_MODEL:-${DEFAULT_BASE_MODEL}}}"
if [[ $# -ge 1 ]]; then
  shift
fi

[[ -s "${BASE_MODEL}/config.json" ]] || {
  echo "ERROR: invalid or incomplete full base model: ${BASE_MODEL}" >&2
  exit 2
}

export LMUData HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true
SUITE_CONFIG="${PROJECT_DIR}/configs/eval/requested_benchmarks.json" \
  bash "${SCRIPT_DIR}/prepare_eval_assets.sh" --offline

# A full base checkpoint is classified as merged/full, so run_eval bypasses
# merge_lora.py. FastVLM is automatically changed from SDPA to eager attention.
exec bash "${SCRIPT_DIR}/run_eval.sh" \
  --checkpoint "${BASE_MODEL}" \
  --suite requested_benchmarks \
  --mode all \
  --attention-backend sdpa \
  --max-new-tokens 128 \
  "$@"
