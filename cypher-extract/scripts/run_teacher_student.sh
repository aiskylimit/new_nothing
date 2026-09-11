#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default first run: all four teachers and only the full-SFT plus normalized
# full-SFT students for every model family. With --phase all, each model is
# inferred with seed 42 immediately after its training finishes.
MODEL_FAMILIES="${MODEL_FAMILIES:-llama3,qwen3,qwen2.5_coder}"
STUDENT_METHODS="${STUDENT_METHODS:-sft}"
STUDENT_SETTINGS="${STUDENT_SETTINGS:-full_finetune,full_finetune_normalized}"
RUN_SETTINGS="${RUN_SETTINGS:-all}"
RUN_PHASE="${RUN_PHASE:-all}"
INFERENCE_SEEDS="${INFERENCE_SEEDS:-42}"
INFERENCE_DATASETS="${INFERENCE_DATASETS:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TRAIN_OVERRIDES=()

SUPPORTED_MODEL_FAMILIES=(
  llama3
  qwen3
  qwen2.5_coder
)

ALL_STUDENT_METHODS=(
  sft
  fkl
  rkl
  sfkl
  srkl
  csd
  hpd
  amid
  fdd_sfkl
  fdd_srkl
  distillm_adaptive_sfkl
  distillm_adaptive_srkl
)

SUPPORTED_SETTINGS=(
  lora
  lora_normalized
  full_finetune
  full_finetune_normalized
)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_teacher_student.sh [options] [-- key=value ...]

Options:
  --families CSV          llama3,qwen3,qwen2.5_coder (default: all three)
  --settings CSV          lora,lora_normalized,full_finetune,
                          full_finetune_normalized, or all (default: all)
  --student-settings CSV  Settings that also train/infer students, or all
                          (default: full_finetune,full_finetune_normalized)
  --student-methods CSV   sft, selected distillation methods, all, or none
                          (default: sft)
  --phase VALUE           train, infer, or all (default: all)
  --seeds CSV             Inference seeds (default: 42)
  --datasets CSV          Optional inference dataset selection
  --skip-completed        Skip completed training/inference work (default)
  --no-skip-completed     Always invoke training, even if outputs are complete
  -h, --help              Show this help

Examples:
  bash scripts/run_teacher_student.sh --families qwen3 --phase train
  bash scripts/run_teacher_student.sh --student-methods all --phase train
  bash scripts/run_teacher_student.sh --families llama3 -- num_train_epochs=1
  bash scripts/run_teacher_student.sh --families qwen3 --no-skip-completed
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --families)
      [[ $# -ge 2 ]] || { echo "--families requires a value" >&2; exit 2; }
      MODEL_FAMILIES="$2"
      shift 2
      ;;
    --families=*) MODEL_FAMILIES="${1#*=}"; shift ;;
    --settings)
      [[ $# -ge 2 ]] || { echo "--settings requires a value" >&2; exit 2; }
      RUN_SETTINGS="$2"
      shift 2
      ;;
    --settings=*) RUN_SETTINGS="${1#*=}"; shift ;;
    --student-settings)
      [[ $# -ge 2 ]] || { echo "--student-settings requires a value" >&2; exit 2; }
      STUDENT_SETTINGS="$2"
      shift 2
      ;;
    --student-settings=*) STUDENT_SETTINGS="${1#*=}"; shift ;;
    --student-methods)
      [[ $# -ge 2 ]] || { echo "--student-methods requires a value" >&2; exit 2; }
      STUDENT_METHODS="$2"
      shift 2
      ;;
    --student-methods=*) STUDENT_METHODS="${1#*=}"; shift ;;
    --phase)
      [[ $# -ge 2 ]] || { echo "--phase requires a value" >&2; exit 2; }
      RUN_PHASE="$2"
      shift 2
      ;;
    --phase=*) RUN_PHASE="${1#*=}"; shift ;;
    --seeds)
      [[ $# -ge 2 ]] || { echo "--seeds requires a value" >&2; exit 2; }
      INFERENCE_SEEDS="$2"
      shift 2
      ;;
    --seeds=*) INFERENCE_SEEDS="${1#*=}"; shift ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "--datasets requires a value" >&2; exit 2; }
      INFERENCE_DATASETS="$2"
      shift 2
      ;;
    --datasets=*) INFERENCE_DATASETS="${1#*=}"; shift ;;
    --skip-completed)
      SKIP_COMPLETED=1
      shift
      ;;
    --no-skip-completed)
      SKIP_COMPLETED=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_OVERRIDES=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use -- before training key=value overrides." >&2
      exit 2
      ;;
  esac
done

case "${RUN_PHASE}" in
  all | train | infer) ;;
  *)
    echo "Unsupported RUN_PHASE=${RUN_PHASE}; expected all, train, or infer." >&2
    exit 2
    ;;
esac

case "${SKIP_COMPLETED}" in
  0 | 1) ;;
  *)
    echo "Unsupported SKIP_COMPLETED=${SKIP_COMPLETED}; expected 0 or 1." >&2
    exit 2
    ;;
esac

if [[ "${RUN_SETTINGS}" == "all" ]]; then
  SELECTED_SETTINGS=("${SUPPORTED_SETTINGS[@]}")
else
  IFS=',' read -r -a SELECTED_SETTINGS <<< "${RUN_SETTINGS}"
  if (( ${#SELECTED_SETTINGS[@]} == 0 )); then
    echo "--settings must select at least one setting." >&2
    exit 2
  fi
  for selected_setting in "${SELECTED_SETTINGS[@]}"; do
    supported=0
    for available_setting in "${SUPPORTED_SETTINGS[@]}"; do
      if [[ "${selected_setting}" == "${available_setting}" ]]; then
        supported=1
        break
      fi
    done
    if (( supported == 0 )); then
      echo "Unsupported run setting: ${selected_setting}" >&2
      exit 2
    fi
  done
fi

IFS=',' read -r -a SELECTED_MODEL_FAMILIES <<< "${MODEL_FAMILIES}"
if (( ${#SELECTED_MODEL_FAMILIES[@]} == 0 )); then
  echo "MODEL_FAMILIES must select at least one model family." >&2
  exit 2
fi
for selected_family in "${SELECTED_MODEL_FAMILIES[@]}"; do
  supported=0
  for available_family in "${SUPPORTED_MODEL_FAMILIES[@]}"; do
    if [[ "${selected_family}" == "${available_family}" ]]; then
      supported=1
      break
    fi
  done
  if (( supported == 0 )); then
    echo "Unsupported model family: ${selected_family}" >&2
    exit 2
  fi
done

if [[ "${STUDENT_METHODS}" == "none" ]]; then
  SELECTED_STUDENT_SETTINGS=()
elif [[ "${STUDENT_SETTINGS}" == "all" ]]; then
  SELECTED_STUDENT_SETTINGS=("${SELECTED_SETTINGS[@]}")
else
  IFS=',' read -r -a REQUESTED_STUDENT_SETTINGS <<< "${STUDENT_SETTINGS}"
  SELECTED_STUDENT_SETTINGS=()
  if (( ${#REQUESTED_STUDENT_SETTINGS[@]} == 0 )); then
    echo "STUDENT_SETTINGS must select at least one setting." >&2
    exit 2
  fi

  for selected_setting in "${REQUESTED_STUDENT_SETTINGS[@]}"; do
    supported=0
    for available_setting in "${SUPPORTED_SETTINGS[@]}"; do
      if [[ "${selected_setting}" == "${available_setting}" ]]; then
        supported=1
        break
      fi
    done
    if (( supported == 0 )); then
      echo "Unsupported student setting: ${selected_setting}" >&2
      exit 2
    fi
    for run_setting in "${SELECTED_SETTINGS[@]}"; do
      if [[ "${selected_setting}" == "${run_setting}" ]]; then
        SELECTED_STUDENT_SETTINGS+=("${selected_setting}")
        break
      fi
    done
  done
fi

if [[ "${STUDENT_METHODS}" == "none" ]]; then
  SELECTED_METHODS=()
elif [[ "${STUDENT_METHODS}" == "all" ]]; then
  SELECTED_METHODS=("${ALL_STUDENT_METHODS[@]}")
else
  IFS=',' read -r -a SELECTED_METHODS <<< "${STUDENT_METHODS}"
  if (( ${#SELECTED_METHODS[@]} == 0 )); then
    echo "STUDENT_METHODS must select at least one method." >&2
    exit 2
  fi

  for selected_method in "${SELECTED_METHODS[@]}"; do
    supported=0
    for available_method in "${ALL_STUDENT_METHODS[@]}"; do
      if [[ "${selected_method}" == "${available_method}" ]]; then
        supported=1
        break
      fi
    done
    if (( supported == 0 )); then
      echo "Unsupported student method: ${selected_method}" >&2
      exit 2
    fi
  done
fi

setting_paths() {
  local model_family="$1"
  local setting="$2"
  local config_family="${model_family}"

  if [[ "${model_family}" == "qwen2.5_coder" ]]; then
    config_family="qwen2.5"
  fi

  case "${setting}" in
    lora)
      if [[ "${model_family}" == "qwen2.5_coder" ]]; then
        CONFIG_DIRECTORY="qwen2.5_coder"
      else
        CONFIG_DIRECTORY="${model_family}"
      fi
      TEACHER_CONFIG="teacher_lora_${model_family}.yaml"
      TEACHER_METHOD="teacher_lora"
      ;;
    lora_normalized)
      CONFIG_DIRECTORY="${config_family}_normalized_loss"
      TEACHER_CONFIG="teacher_lora_${model_family}_normalized_loss.yaml"
      TEACHER_METHOD="teacher_lora"
      ;;
    full_finetune)
      CONFIG_DIRECTORY="${config_family}_full_finetune"
      TEACHER_CONFIG="teacher_full_${model_family}.yaml"
      TEACHER_METHOD="teacher_full"
      ;;
    full_finetune_normalized)
      CONFIG_DIRECTORY="${config_family}_full_finetune_normalized_loss"
      TEACHER_CONFIG="teacher_full_${model_family}_normalized_loss.yaml"
      TEACHER_METHOD="teacher_full"
      ;;
  esac
}

has_student_setting() {
  local candidate="$1"
  local selected_setting
  for selected_setting in "${SELECTED_STUDENT_SETTINGS[@]}"; do
    if [[ "${candidate}" == "${selected_setting}" ]]; then
      return 0
    fi
  done
  return 1
}

# Validate every selected config before starting a long training run.
if [[ "${RUN_PHASE}" == "all" || "${RUN_PHASE}" == "train" ]]; then
  for model_family in "${SELECTED_MODEL_FAMILIES[@]}"; do
    for setting in "${SELECTED_SETTINGS[@]}"; do
      setting_paths "${model_family}" "${setting}"
      teacher_path="${PROJECT_ROOT}/configs/distillation/${TEACHER_CONFIG}"
      if [[ ! -f "${teacher_path}" ]]; then
        echo "Missing teacher config: ${teacher_path}" >&2
        exit 2
      fi

      if has_student_setting "${setting}"; then
        for method in "${SELECTED_METHODS[@]}"; do
          student_path="${PROJECT_ROOT}/configs/${CONFIG_DIRECTORY}/${method}.yaml"
          if [[ ! -f "${student_path}" ]]; then
            echo "Missing student config: ${student_path}" >&2
            exit 2
          fi
        done
      fi
    done
  done
fi

cd "${PROJECT_ROOT}"

training_output_dir() {
  local config_path="$1"
  local output_dir=""
  local override

  # Command-line overrides have the same precedence as the training CLI.
  for override in "${TRAIN_OVERRIDES[@]}"; do
    if [[ "${override}" == output_dir=* ]]; then
      output_dir="${override#output_dir=}"
    fi
  done

  if [[ -z "${output_dir}" ]]; then
    output_dir="$(
      sed -n 's/^[[:space:]]*output_dir:[[:space:]]*//p' "${config_path}" \
        | tail -n 1 \
        | sed -e 's/[[:space:]]*$//'
    )"
    output_dir="${output_dir#\"}"
    output_dir="${output_dir%\"}"
    output_dir="${output_dir#\'}"
    output_dir="${output_dir%\'}"
  fi

  if [[ -z "${output_dir}" ]]; then
    echo "Could not resolve output_dir from ${config_path}" >&2
    return 2
  fi

  if [[ "${output_dir}" == /* ]]; then
    printf '%s\n' "${output_dir}"
  else
    printf '%s/%s\n' "${PROJECT_ROOT}" "${output_dir#./}"
  fi
}

has_final_model_weights() {
  local output_dir="$1"
  local filename
  local weight_files=(
    adapter_model.safetensors
    adapter_model.bin
    model.safetensors
    model.safetensors.index.json
    pytorch_model.bin
    pytorch_model.bin.index.json
  )

  for filename in "${weight_files[@]}"; do
    if [[ -s "${output_dir}/${filename}" ]]; then
      return 0
    fi
  done
  return 1
}

training_is_complete() {
  local output_dir="$1"

  # These root-level files are written only after Trainer.train() returns,
  # the final model is saved, and metrics/state are published successfully.
  [[ -s "${output_dir}/train_results.json" ]] || return 1
  [[ -s "${output_dir}/trainer_state.json" ]] || return 1
  has_final_model_weights "${output_dir}"
}

run_training() {
  local label="$1"
  local config_path="$2"
  local output_dir

  output_dir="$(training_output_dir "${config_path}")"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && training_is_complete "${output_dir}"; then
    echo "Training already completed; skipping ${label}"
    echo "Output: ${output_dir}"
    return 0
  fi

  bash scripts/train.sh "${config_path}" "${TRAIN_OVERRIDES[@]}"
}

run_inference() {
  local model_family="$1"
  local setting="$2"
  local methods="$3"
  local inference_args=(
    --methods "${methods}"
    --seeds "${INFERENCE_SEEDS}"
  )

  if [[ -n "${INFERENCE_DATASETS}" ]]; then
    inference_args+=(--datasets "${INFERENCE_DATASETS}")
  fi

  echo
  echo "============================================================"
  echo "Running ${model_family} inference: ${setting}"
  echo "Methods: ${methods}"
  echo "Seeds: ${INFERENCE_SEEDS}"
  echo "============================================================"
  bash scripts/infer_all.sh "${model_family}" "${setting}" "${inference_args[@]}"
}

if [[ "${RUN_PHASE}" == "all" || "${RUN_PHASE}" == "train" ]]; then
  for model_family in "${SELECTED_MODEL_FAMILIES[@]}"; do
    for setting in "${SELECTED_SETTINGS[@]}"; do
      setting_paths "${model_family}" "${setting}"

      echo
      echo "============================================================"
      echo "Training ${model_family} teacher: ${setting}"
      echo "Config: configs/distillation/${TEACHER_CONFIG}"
      echo "============================================================"
      run_training \
        "${model_family} teacher: ${setting}" \
        "configs/distillation/${TEACHER_CONFIG}"
      if [[ "${RUN_PHASE}" == "all" ]]; then
        run_inference "${model_family}" "${setting}" "${TEACHER_METHOD}"
      fi

      if has_student_setting "${setting}"; then
        for method in "${SELECTED_METHODS[@]}"; do
          echo
          echo "============================================================"
          echo "Training ${model_family} student: ${setting}/${method}"
          echo "Config: configs/${CONFIG_DIRECTORY}/${method}.yaml"
          echo "============================================================"
          run_training \
            "${model_family} student: ${setting}/${method}" \
            "configs/${CONFIG_DIRECTORY}/${method}.yaml"
          if [[ "${RUN_PHASE}" == "all" ]]; then
            run_inference "${model_family}" "${setting}" "${method}"
          fi
        done
      fi
    done
  done
fi

if [[ "${RUN_PHASE}" == "infer" ]]; then
  for model_family in "${SELECTED_MODEL_FAMILIES[@]}"; do
    for setting in "${SELECTED_SETTINGS[@]}"; do
      setting_paths "${model_family}" "${setting}"
      inference_methods="${TEACHER_METHOD}"
      if has_student_setting "${setting}"; then
        for method in "${SELECTED_METHODS[@]}"; do
          inference_methods+=",${method}"
        done
      fi

      run_inference "${model_family}" "${setting}" "${inference_methods}"
    done
  done
fi

echo
echo "Teacher/student ${RUN_PHASE} pipeline completed successfully."
