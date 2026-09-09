#!/usr/bin/env bash
# Run the Qwen3-VL-4B teacher / FastVLM-0.5B student baselines.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

runner="run_qwen3_teacher_4b_fastvlm_student_05b.sh"
runner_path="${SCRIPT_DIR}/${runner}"

[[ -f "${runner_path}" ]] || { echo "Missing script: ${runner_path}" >&2; exit 1; }

printf '\n=== [%s] Starting %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"
bash "${runner_path}"
printf '=== [%s] Finished %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${runner}"

printf '\nAll Qwen3-VL-4B / FastVLM-0.5B baselines completed successfully.\n'
