#!/usr/bin/env bash
# Run all baselines sequentially for Qwen2.5-VL-7B teacher and Qwen2-VL-2B student.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PAIR_DIR="${SCRIPT_DIR}/qwen25_teacher_7b_qwen2_student_2b"
export PROJECT_DIR

SCRIPTS=(
  "train_qwen25_teacher_7b_qwen2_student_2b_ce_only.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_cgkd.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_dskd_v2_with_eta.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_dwa_kd.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_emkd.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_mcw_kd.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_scva.sh"
  "train_qwen25_teacher_7b_qwen2_student_2b_sre.sh"
)

for script_name in "${SCRIPTS[@]}"; do
  script_path="${PAIR_DIR}/${script_name}"
  [[ -f "${script_path}" ]] || { echo "Missing script: ${script_path}" >&2; exit 1; }
  printf '\n[%s] Starting %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
  bash "${script_path}"
  printf '[%s] Finished %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
done

printf '\nAll baselines completed successfully for qwen25_teacher_7b_qwen2_student_2b.\n'
