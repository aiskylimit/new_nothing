#!/usr/bin/env bash
# Run SCVA, CGKD, and SCVA-CGKD for Qwen3-VL-8B teacher and Qwen2.5-VL-3B student.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RUNNERS=(
  "scva/train_qwen3_teacher_8b_qwen25_student_3b_scva.sh"
  "cgkd/train_qwen3_teacher_8b_qwen25_student_3b_cgkd.sh"
  "scva_cgkd/train_qwen3_teacher_8b_qwen25_student_3b_scva_cgkd.sh"
)

for runner in "${RUNNERS[@]}"; do
  runner_path="${SCRIPT_DIR}/${runner}"
  [[ -f "${runner_path}" ]] || { echo "Missing script: ${runner_path}" >&2; exit 1; }
  printf '\n=== [%s] Starting %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"
  bash "${runner_path}"
  printf '=== [%s] Finished %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"
done

printf '\nSCVA, CGKD, and SCVA-CGKD completed successfully for Qwen3-VL-8B / Qwen2.5-VL-3B.\n'
