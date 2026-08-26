#!/usr/bin/env bash
# Run all baselines sequentially for Qwen3-VL-8B teacher and Qwen2.5-VL-3B student.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PAIR_DIR="${SCRIPT_DIR}/qwen3_teacher_8b_qwen25_student_3b"
export PROJECT_DIR

SCRIPTS=(
  "train_qwen3_teacher_8b_qwen25_student_3b_ce_only.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_dskd_v2_with_eta.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_dwa_kd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_emkd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_mcw_kd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_sre.sh"
)

for script_name in "${SCRIPTS[@]}"; do
  script_path="${PAIR_DIR}/${script_name}"
  [[ -f "${script_path}" ]] || { echo "Missing script: ${script_path}" >&2; exit 1; }
  printf '\n[%s] Starting %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
  bash "${script_path}"
  printf '[%s] Finished %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
done

printf '\nAll baselines completed successfully for qwen3_teacher_8b_qwen25_student_3b.\n'
