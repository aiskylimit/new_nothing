#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
EVAL_ENV=/mnt/local/uvenvs/vlm-distill-eval-2/bin/activate
SUITE_CONFIG="${PROJECT_DIR}/configs/eval/requested_benchmarks_no_mme_ocrbench.json"
MODELS_ROOT=/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models
QWEN2VL_2B_BASE="${MODELS_ROOT}/Qwen/Qwen2-VL-2B-Instruct"
QWEN2VL_7B_BASE="${MODELS_ROOT}/Qwen/Qwen2-VL-7B-Instruct"

adapter_sources=(
  "${MODELS_ROOT}/qwen2vl-2b-kd-ce-only-full"
  "${MODELS_ROOT}/qwen2vl-2b-kd-emkd-full"
  "${MODELS_ROOT}/qwen2vl-2b-kd-sre-full"
)
adapter_gpus=(4 5 6)

source "${EVAL_ENV}"
cd "${PROJECT_DIR}"

export LMUData="${LMUData:-${PROJECT_DIR}/eval_data/LMUData}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=true

[[ -s "${QWEN2VL_2B_BASE}/config.json" ]] || {
  echo "ERROR: invalid or incomplete Qwen2-VL-2B base model: ${QWEN2VL_2B_BASE}" >&2
  exit 2
}
[[ -s "${QWEN2VL_7B_BASE}/config.json" ]] || {
  echo "ERROR: invalid or incomplete Qwen2-VL-7B base model: ${QWEN2VL_7B_BASE}" >&2
  exit 2
}

# Accept either an adapter checkpoint itself or a run directory containing
# checkpoint-N directories. In the latter case, use the largest numeric step.
resolve_adapter_checkpoint() {
  local source_dir="$1"
  local candidate checkpoint_name checkpoint_step
  local selected=""
  local latest_step=-1

  [[ -d "${source_dir}" ]] || {
    echo "ERROR: adapter directory does not exist: ${source_dir}" >&2
    return 2
  }

  if [[ -s "${source_dir}/adapter_config.json" ]]; then
    printf '%s\n' "${source_dir}"
    return 0
  fi

  shopt -s nullglob
  for candidate in "${source_dir}"/checkpoint-*; do
    [[ -d "${candidate}" && -s "${candidate}/adapter_config.json" ]] || continue
    checkpoint_name="${candidate##*/}"
    checkpoint_step="${checkpoint_name#checkpoint-}"
    [[ "${checkpoint_step}" =~ ^[0-9]+$ ]] || continue
    if (( checkpoint_step > latest_step )); then
      latest_step="${checkpoint_step}"
      selected="${candidate}"
    fi
  done
  shopt -u nullglob

  [[ -n "${selected}" ]] || {
    echo "ERROR: no valid adapter checkpoint found under: ${source_dir}" >&2
    return 2
  }
  printf '%s\n' "${selected}"
}

adapter_checkpoints=()
for source_dir in "${adapter_sources[@]}"; do
  adapter_checkpoints+=("$(resolve_adapter_checkpoint "${source_dir}")")
done

# Asset preparation writes a shared manifest, so finish it once before the four
# concurrent evaluations start.
SUITE_CONFIG="${SUITE_CONFIG}" \
  bash "${PROJECT_DIR}/scripts/eval/prepare_eval_assets.sh" --offline

run_adapter() {
  local gpu="$1"
  local checkpoint="$2"

  printf '[%s] GPU %s: evaluating LoRA %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${gpu}" "${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${PROJECT_DIR}/scripts/eval/run_eval.sh" \
      --checkpoint "${checkpoint}" \
      --base-model "${QWEN2VL_2B_BASE}" \
      --suite requested_benchmarks \
      --suite-config "${SUITE_CONFIG}" \
      --mode all \
      --attention-backend sdpa \
      --max-new-tokens 128
}

run_base_model() {
  local gpu="$1"

  printf '[%s] GPU %s: evaluating base model %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${gpu}" "${QWEN2VL_7B_BASE}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${PROJECT_DIR}/scripts/eval/run_eval.sh" \
      --checkpoint "${QWEN2VL_7B_BASE}" \
      --suite requested_benchmarks \
      --suite-config "${SUITE_CONFIG}" \
      --mode all \
      --attention-backend sdpa \
      --max-new-tokens 128
}

pids=()
labels=()
for index in "${!adapter_checkpoints[@]}"; do
  run_adapter "${adapter_gpus[index]}" "${adapter_checkpoints[index]}" &
  pids+=("$!")
  labels+=("${adapter_sources[index]##*/} (GPU ${adapter_gpus[index]})")
done

run_base_model 7 &
pids+=("$!")
labels+=("Qwen2-VL-7B-Instruct (GPU 7)")

printf 'Started all 4 evaluations concurrently. Results: %s\n' \
  "${PROJECT_DIR}/outputs/eval/requested_benchmarks"

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    printf '[DONE] %s\n' "${labels[index]}"
  else
    status=$?
    printf '[FAILED: exit %s] %s\n' "${status}" "${labels[index]}" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "One or more evaluations failed; inspect their summary.json files for details." >&2
  exit 1
fi

echo "All evaluations completed. Collect them with:"
echo "  bash ${PROJECT_DIR}/project_commands_collect_eval_summary.sh"
