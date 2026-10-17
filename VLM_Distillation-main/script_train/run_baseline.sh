#!/usr/bin/env bash
# Run the baselines for the Qwen2-VL-7B teacher / FastVLM-0.5B student pair.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

runner="run_qwen2_teacher_7b_fastvlm_student_05b.sh"
runner_path="${SCRIPT_DIR}/${runner}"

[[ -f "${runner_path}" ]] || { echo "Missing script: ${runner_path}" >&2; exit 1; }

printf '\n=== [%s] Starting %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"
bash "${runner_path}"
printf '=== [%s] Finished %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"

printf '\nAll Qwen2-VL-7B / FastVLM-0.5B baselines completed successfully.\n'
