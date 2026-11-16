#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
EVAL_ENV=/mnt/local/uvenvs/vlm-distill-eval-2/bin/activate
SUITE_CONFIG="${PROJECT_DIR}/configs/eval/requested_benchmarks_no_mme_ocrbench.json"
BASE_MODEL=/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-7B-Instruct

source "${EVAL_ENV}"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=7
export LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true

[[ -s "${BASE_MODEL}/config.json" ]] || {
  echo "ERROR: invalid or incomplete Qwen2.5-VL-7B base model: ${BASE_MODEL}" >&2
  exit 2
}

# Prepare the nine-dataset suite (MME and OCRBench are excluded).
SUITE_CONFIG="${SUITE_CONFIG}" \
  bash "${PROJECT_DIR}/scripts/eval/prepare_eval_assets.sh" --offline

printf '[%s] GPU 7: evaluating base model %s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${BASE_MODEL}"

exec bash "${PROJECT_DIR}/scripts/eval/run_eval.sh" \
  --checkpoint "${BASE_MODEL}" \
  --suite requested_benchmarks \
  --suite-config "${SUITE_CONFIG}" \
  --mode all \
  --attention-backend sdpa \
  --max-new-tokens 128
