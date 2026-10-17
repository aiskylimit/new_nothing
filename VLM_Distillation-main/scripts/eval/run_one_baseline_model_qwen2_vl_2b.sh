#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BASE_MODEL="${BASE_MODEL:-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen2-VL-2B-Instruct}"
OUTPUT_DIR="${BASE_MODEL_EVAL_ROOT:-${PROJECT_DIR}/outputs/eval/base_models}/qwen2_vl_2b"

exec bash "${SCRIPT_DIR}/run_one_baseline_model.sh" "${BASE_MODEL}" \
  --model-name qwen2_vl_2b \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
