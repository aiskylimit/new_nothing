#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"

# Defaults for this machine. Override either value through the environment or
# pass CHECKPOINT and BASE_MODEL as the first two positional arguments.
DEFAULT_CHECKPOINT="${PROJECT_DIR}/outputs/stress_maxlen_20/emkd/checkpoint-20"
DEFAULT_BASE_MODEL="/home/vinhld8/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
CHECKPOINT="${1:-${TRAINED_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
BASE_MODEL="${2:-${BASE_MODEL:-${DEFAULT_BASE_MODEL}}}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-10}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  set --
fi

for dataset in AI2D_TEST_NO_MASK GQA_TestDev_Balanced; do
  [[ -s "${LMUData}/${dataset}.tsv" ]] || {
    echo "ERROR: missing local dataset: ${LMUData}/${dataset}.tsv" >&2
    exit 2
  }
done
[[ -s "${CHECKPOINT}/adapter_config.json" ]] || {
  echo "ERROR: invalid LoRA checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
[[ -s "${BASE_MODEL}/config.json" ]] || {
  echo "ERROR: invalid local base model: ${BASE_MODEL}" >&2
  exit 2
}
[[ "${SMOKE_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: SMOKE_SAMPLES must be a positive integer" >&2
  exit 2
}

export LMUData HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true
SUITE_CONFIG="${PROJECT_DIR}/configs/eval/ai2d_gqa_smoke.json" \
  bash "${SCRIPT_DIR}/prepare_eval_assets.sh" --offline

exec bash "${SCRIPT_DIR}/run_eval.sh" \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --suite ai2d_gqa_smoke \
  --mode all \
  --attention-backend sdpa \
  --max-new-tokens 32 \
  --max-samples "${SMOKE_SAMPLES}" \
  "$@"
