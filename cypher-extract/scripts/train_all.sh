#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/train_all.sh <model-family> <setting> [key=value ...]" >&2
  echo "Model families: qwen3, llama3, qwen2.5_coder" >&2
  echo "Settings: lora, lora_normalized, full_finetune, full_finetune_normalized" >&2
  exit 2
fi

MODEL_FAMILY="$1"
SETTING="$2"
shift 2

case "${MODEL_FAMILY}" in
  qwen3)
    CONFIG_FAMILY="qwen3"
    FAMILY_LABEL="Qwen3"
    ;;
  llama3)
    CONFIG_FAMILY="llama3"
    FAMILY_LABEL="Llama3"
    ;;
  qwen2.5_coder)
    CONFIG_FAMILY="qwen2.5"
    FAMILY_LABEL="Qwen2.5-Coder"
    ;;
  *)
    echo "Unsupported model family: ${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

case "${SETTING}" in
  lora)
    if [[ "${MODEL_FAMILY}" == "qwen2.5_coder" ]]; then
      CONFIG_DIRECTORY="qwen2.5_coder"
    else
      CONFIG_DIRECTORY="${MODEL_FAMILY}"
    fi
    TEACHER_KIND="teacher_lora"
    TEACHER_CONFIG_NAME="teacher_lora_${MODEL_FAMILY}.yaml"
    TEACHER_OVERRIDE_KEY="ref_model_adapters"
    DEFAULT_RESULTS_SUBDIR="results/lora"
    ;;
  lora_normalized)
    CONFIG_DIRECTORY="${CONFIG_FAMILY}_normalized_loss"
    TEACHER_KIND="teacher_lora"
    TEACHER_CONFIG_NAME="teacher_lora_${MODEL_FAMILY}_normalized_loss.yaml"
    TEACHER_OVERRIDE_KEY="ref_model_adapters"
    DEFAULT_RESULTS_SUBDIR="results/lora_normalized"
    ;;
  full_finetune)
    CONFIG_DIRECTORY="${CONFIG_FAMILY}_full_finetune"
    TEACHER_KIND="teacher_full"
    TEACHER_CONFIG_NAME="teacher_full_${MODEL_FAMILY}.yaml"
    TEACHER_OVERRIDE_KEY="ref_model"
    DEFAULT_RESULTS_SUBDIR="results/full_finetune"
    ;;
  full_finetune_normalized)
    CONFIG_DIRECTORY="${CONFIG_FAMILY}_full_finetune_normalized_loss"
    TEACHER_KIND="teacher_full"
    TEACHER_CONFIG_NAME="teacher_full_${MODEL_FAMILY}_normalized_loss.yaml"
    TEACHER_OVERRIDE_KEY="ref_model"
    DEFAULT_RESULTS_SUBDIR="results/full_finetune_normalized"
    ;;
  *)
    echo "Unsupported training setting: ${SETTING}" >&2
    exit 2
    ;;
esac

CONFIG_DIR="${PROJECT_ROOT}/configs/${CONFIG_DIRECTORY}"
TEACHER_CONFIG="${PROJECT_ROOT}/configs/distillation/${TEACHER_CONFIG_NAME}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/${DEFAULT_RESULTS_SUBDIR}}"
if [[ "${RESULTS_ROOT}" != /* ]]; then
  RESULTS_ROOT="${PROJECT_ROOT}/${RESULTS_ROOT}"
fi
FAMILY_RESULTS="${RESULTS_ROOT}/${MODEL_FAMILY}"
TEACHER_OUTPUT="${FAMILY_RESULTS}/${TEACHER_KIND}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TRAIN_ALL_LOG_DIR:-${FAMILY_RESULTS}/run_all_logs/${RUN_ID}}"

CONFIG_NAMES=(
  sft.yaml
  fkl.yaml
  rkl.yaml
  sfkl.yaml
  srkl.yaml
  csd.yaml
  hpd.yaml
  amid.yaml
  fdd_sfkl.yaml
  fdd_srkl.yaml
  distillm_adaptive_sfkl.yaml
  distillm_adaptive_srkl.yaml
)

for override in "$@"; do
  if [[ "${override}" == resume_from_checkpoint=* ]]; then
    echo "Do not resume through train_all: one checkpoint cannot be applied to every method." >&2
    echo "Resume one run with scripts/train.sh and its matching config/output_dir." >&2
    exit 2
  fi
done

if [[ ! -f "${TEACHER_CONFIG}" ]]; then
  echo "Missing teacher config: ${TEACHER_CONFIG}" >&2
  exit 2
fi
for config_name in "${CONFIG_NAMES[@]}"; do
  if [[ ! -f "${CONFIG_DIR}/${config_name}" ]]; then
    echo "Missing config: ${CONFIG_DIR}/${config_name}" >&2
    exit 2
  fi
done

fresh_outputs=("${TEACHER_OUTPUT}")
for config_name in "${CONFIG_NAMES[@]}"; do
  fresh_outputs+=("${FAMILY_RESULTS}/${config_name%.yaml}")
done
for output_dir in "${fresh_outputs[@]}"; do
  for checkpoint in "${output_dir}"/checkpoint-*; do
    if [[ -d "${checkpoint}" ]]; then
      echo "Refusing fresh train-all run: old checkpoint found at ${checkpoint}" >&2
      echo "Choose a new RESULTS_ROOT. Resume a single run through scripts/train.sh instead." >&2
      exit 2
    fi
  done
done

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

teacher_config_path="configs/distillation/${TEACHER_CONFIG_NAME}"
teacher_log_path="${LOG_DIR}/${TEACHER_CONFIG_NAME%.yaml}.log"
echo
echo "============================================================"
echo "Running ${teacher_config_path}"
echo "Output: ${TEACHER_OUTPUT}"
echo "Log: ${teacher_log_path}"
echo "============================================================"

if bash scripts/train.sh "${teacher_config_path}" "$@" \
  "output_dir=${TEACHER_OUTPUT}" 2>&1 | tee "${teacher_log_path}"; then
  echo "Completed: ${TEACHER_CONFIG_NAME%.yaml}"
else
  status=${PIPESTATUS[0]}
  echo "Failed: ${TEACHER_CONFIG_NAME%.yaml} (exit ${status}); dependent runs were not started." >&2
  exit "${status}"
fi

has_full_model_weights() {
  compgen -G "${TEACHER_OUTPUT}/model*.safetensors" > /dev/null ||
    compgen -G "${TEACHER_OUTPUT}/pytorch_model*.bin" > /dev/null
}

if [[ "${TEACHER_KIND}" == "teacher_lora" ]]; then
  if [[ ! -f "${TEACHER_OUTPUT}/adapter_config.json" ]] || \
     [[ ! -f "${TEACHER_OUTPUT}/adapter_model.safetensors" && ! -f "${TEACHER_OUTPUT}/adapter_model.bin" ]]; then
    echo "Teacher run completed but did not create LoRA adapter weights in ${TEACHER_OUTPUT}" >&2
    exit 1
  fi
elif [[ ! -f "${TEACHER_OUTPUT}/config.json" ]] || ! has_full_model_weights; then
  echo "Teacher run completed but did not create full-model weights in ${TEACHER_OUTPUT}" >&2
  exit 1
fi

failed=()
for config_name in "${CONFIG_NAMES[@]}"; do
  method="${config_name%.yaml}"
  config_path="configs/${CONFIG_DIRECTORY}/${config_name}"
  log_path="${LOG_DIR}/${method}.log"
  output_dir="${FAMILY_RESULTS}/${method}"
  method_overrides=("output_dir=${output_dir}")
  if [[ "${method}" != "sft" ]]; then
    method_overrides+=("${TEACHER_OVERRIDE_KEY}=${TEACHER_OUTPUT}")
  fi

  echo
  echo "============================================================"
  echo "Running ${config_path}"
  echo "Output: ${output_dir}"
  echo "Log: ${log_path}"
  echo "============================================================"

  if bash scripts/train.sh "${config_path}" "$@" "${method_overrides[@]}" 2>&1 | tee "${log_path}"; then
    echo "Completed: ${method}"
  else
    status=${PIPESTATUS[0]}
    failed+=("${method}:${status}")
    echo "Failed: ${method} (exit ${status})" >&2
    if [[ "${CONTINUE_ON_ERROR:-0}" != "1" ]]; then
      echo "Set CONTINUE_ON_ERROR=1 to continue after a failed run." >&2
      exit "${status}"
    fi
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "Failed runs: ${failed[*]}" >&2
  exit 1
fi

echo
echo "All ${FAMILY_LABEL} ${SETTING} configs completed successfully."
echo "Logs: ${LOG_DIR}"
