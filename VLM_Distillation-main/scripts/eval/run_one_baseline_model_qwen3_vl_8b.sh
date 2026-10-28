#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BASE_MODEL="${BASE_MODEL:-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${BASE_MODEL_EVAL_ROOT:-${PROJECT_DIR}/outputs/eval/base_models}/qwen3_vl_8b"

exec bash "${SCRIPT_DIR}/run_one_baseline_model.sh" "${BASE_MODEL}" \
  --model-name qwen3_vl_8b \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
