#!/usr/bin/env bash
# Run all baseline pairs sequentially by invoking the existing per-pair runners:
#   1) run_qwen3_teacher_4b_fastvlm_student_05b.sh
#   2) run_qwen3_teacher_8b_qwen25_student_3b.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUNNERS=(
  # "run_qwen3_teacher_4b_fastvlm_student_05b.sh"
  "run_qwen3_teacher_8b_qwen25_student_3b.sh"
)

for runner in "${RUNNERS[@]}"; do
  runner_path="${SCRIPT_DIR}/${runner}"
  [[ -f "${runner_path}" ]] || { echo "Missing script: ${runner_path}" >&2; exit 1; }
  printf '\n=== [%s] Starting %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"
  bash "${runner_path}"
  printf '=== [%s] Finished %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"
done

printf '\nAll baseline pairs completed successfully.\n'
