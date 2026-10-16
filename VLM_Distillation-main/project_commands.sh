#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
DOWNLOAD_ROOT=/mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main
DOWNLOAD_DATA_DIR="${DOWNLOAD_ROOT}/train_data"
DATA_DIR="${PROJECT_DIR}/train_data"
TRAIN_ENV=/mnt/local/uvenvs/vlm-distill/bin/activate
TRAIN_DIR="${PROJECT_DIR}/script_train/qwen2_teacher_7b_fastvlm_student_05b"
EVAL_SCRIPT="${PROJECT_DIR}/project_commands_eval.sh"
BASE_MODEL="${BASE_MODEL:-${DOWNLOAD_ROOT}/models/KamilaMila/FastVLM-0.5B}"

source "${TRAIN_ENV}"
cd "${PROJECT_DIR}"

# Copy metadata into the relative train_data tree.  cmp also safely handles
# the case where the download directory and code directory are the same.
mkdir -p "${DATA_DIR}/ocr_vqa"
if ! cmp -s "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" "${DATA_DIR}/llava_v1_5_mix665k.json"; then
  cp -f "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" "${DATA_DIR}/llava_v1_5_mix665k.json"
fi
if ! cmp -s "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" "${DATA_DIR}/ocr_vqa/dataset.json"; then
  cp -f "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" "${DATA_DIR}/ocr_vqa/dataset.json"
fi

# Training uses IMAGE_DIR=train_data. The archives are expected to have
# already been extracted under this directory.
# unzip -q -o "${DOWNLOAD_DATA_DIR}/coco/train2017.zip" -d "${DATA_DIR}/coco"
# unzip -q -o "${DOWNLOAD_DATA_DIR}/gqa/images.zip" -d "${DATA_DIR}/gqa"
# unzip -q -o "${DOWNLOAD_DATA_DIR}/textvqa/train_val_images.zip" -d "${DATA_DIR}/textvqa"
# unzip -q -o "${DOWNLOAD_DATA_DIR}/ocr_vqa/ocr_vqa_images.zip" -d "${DATA_DIR}/ocr_vqa"
# unzip -q -o "${DOWNLOAD_DATA_DIR}/vg/images.zip" -d "${DATA_DIR}/vg"
# unzip -q -o "${DOWNLOAD_DATA_DIR}/vg/images2.zip" -d "${DATA_DIR}/vg"

# bash download_datatrain.sh

# CE-only training has already completed. Resume from its evaluation, then run
# every remaining training job on GPUs 4,5,6,7 and immediately evaluate its
# newest checkpoint on GPU 4.
CE_RUN_NAME="qwen2_teacher_7b_fastvlm_student_05b_ce_only"
CE_OUTPUT_DIR="${PROJECT_DIR}/outputs/${CE_RUN_NAME}"

printf '\n=== [%s] RESUME WITH EVAL %s on GPU 4 ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${CE_RUN_NAME}"
CUDA_VISIBLE_DEVICES=4 bash "${EVAL_SCRIPT}" "${CE_OUTPUT_DIR}" "${BASE_MODEL}"
printf '=== [%s] COMPLETED eval: %s ===\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${CE_RUN_NAME}"

JOBS=(
  "train_qwen2_teacher_7b_fastvlm_student_05b_dskd_v2_with_eta.sh|qwen2_teacher_7b_fastvlm_student_05b_dskd_v2_with_eta"
  "train_qwen2_teacher_7b_fastvlm_student_05b_dwa_kd.sh|qwen2_teacher_7b_fastvlm_student_05b_dwa_kd"
  "train_qwen2_teacher_7b_fastvlm_student_05b_emkd.sh|qwen2_teacher_7b_fastvlm_student_05b_emkd"
  "train_qwen2_teacher_7b_fastvlm_student_05b_mcw_kd.sh|qwen2_teacher_7b_fastvlm_student_05b_mcw_kd"
  "train_qwen2_teacher_7b_fastvlm_student_05b_sre.sh|qwen2_teacher_7b_fastvlm_student_05b_sre"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r train_script run_name <<< "${job}"
  train_path="${TRAIN_DIR}/${train_script}"
  output_dir="${PROJECT_DIR}/outputs/${run_name}"

  [[ -f "${train_path}" ]] || { echo "ERROR: missing training script: ${train_path}" >&2; exit 1; }

  printf '\n=== [%s] TRAIN %s on GPUs 4,5,6,7 ===\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"
  CUDA_VISIBLE_DEVICES=4,5,6,7 bash "${train_path}"

  printf '\n=== [%s] EVAL %s on GPU 4 ===\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"
  CUDA_VISIBLE_DEVICES=4 bash "${EVAL_SCRIPT}" "${output_dir}" "${BASE_MODEL}"

  printf '=== [%s] COMPLETED train + eval: %s ===\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"
done

printf '\nAll train/eval jobs completed successfully.\n'
