#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-${PROJECT_DIR}/models}"
LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"

datasets=(
  GQA_TestDev_Balanced MME RealWorldQA ScienceQA_TEST AI2D_TEST_NO_MASK
  MMMU_DEV_VAL MMStar ChartQA_TEST DocVQA_VAL TextVQA_VAL OCRBench
)
for dataset in "${datasets[@]}"; do
  [[ -s "${LMUData}/${dataset}.tsv" ]] || {
    echo "ERROR: missing dataset: ${LMUData}/${dataset}.tsv" >&2
    echo "Download every entry from download_evaldata.txt before submitting eval." >&2
    exit 2
  }
done

fastvlm="${MODELS_ROOT}/KamilaMila/FastVLM-0.5B"
qwen2="${MODELS_ROOT}/Qwen/Qwen2-VL-2B-Instruct"
qwen25="${MODELS_ROOT}/Qwen/Qwen2.5-VL-3B-Instruct"
for model in "${fastvlm}" "${qwen2}" "${qwen25}"; do
  if [[ ! -s "${model}/config.json" ]] || \
     [[ ! -s "${model}/tokenizer_config.json" ]] || \
     { [[ ! -s "${model}/processor_config.json" ]] && [[ ! -s "${model}/preprocessor_config.json" ]]; } || \
     { [[ ! -s "${model}/model.safetensors" ]] && [[ ! -s "${model}/model.safetensors.index.json" ]] && \
       [[ ! -s "${model}/pytorch_model.bin" ]] && [[ ! -s "${model}/pytorch_model.bin.index.json" ]]; }; then
    echo "ERROR: missing local base model: ${model}" >&2
    echo "Download every model entry from download_evaldata.txt before submitting eval." >&2
    exit 2
  fi
done

export LMUData HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true
SUITE_CONFIG="${PROJECT_DIR}/configs/eval/requested_benchmarks.json" \
  bash "${SCRIPT_DIR}/prepare_eval_assets.sh" --offline

exec bash "${SCRIPT_DIR}/run_all_methods.sh" \
  --suite requested_benchmarks \
  --attention-backend sdpa \
  --max-new-tokens 128 \
  --base-model-override "KamilaMila/FastVLM-0.5B=${fastvlm}" \
  --base-model-override "Qwen/Qwen2-VL-2B-Instruct=${qwen2}" \
  --base-model-override "Qwen/Qwen2.5-VL-3B-Instruct=${qwen25}" \
  "$@"
