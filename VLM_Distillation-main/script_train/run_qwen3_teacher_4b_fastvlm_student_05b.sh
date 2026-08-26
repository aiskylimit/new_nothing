#!/usr/bin/env bash
# Run all baselines sequentially for Qwen3-VL-4B teacher and FastVLM-0.5B student.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PAIR_DIR="${SCRIPT_DIR}/qwen3_teacher_4b_fastvlm_student_05b"
export PROJECT_DIR

SCRIPTS=(
  "train_qwen3_teacher_4b_fastvlm_student_05b_ce_only.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_dskd_v2_with_eta.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_dwa_kd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_emkd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_mcw_kd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_sre.sh"
)

for script_name in "${SCRIPTS[@]}"; do
  script_path="${PAIR_DIR}/${script_name}"
  [[ -f "${script_path}" ]] || { echo "Missing script: ${script_path}" >&2; exit 1; }
  printf '\n[%s] Starting %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
  bash "${script_path}"
  printf '[%s] Finished %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
done

printf '\nAll baselines completed successfully for qwen3_teacher_4b_fastvlm_student_05b.\n'
