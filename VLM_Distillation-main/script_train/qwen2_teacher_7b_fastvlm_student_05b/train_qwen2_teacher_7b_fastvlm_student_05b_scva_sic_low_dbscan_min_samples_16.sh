#!/usr/bin/env bash
# Joint SCVA + SIC low, with DBSCAN min_samples ablated from 8 to 16.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low_dbscan_min_samples_16}"
export SCVA_DBSCAN_MIN_SAMPLES="${SCVA_DBSCAN_MIN_SAMPLES:-16}"

exec bash "${SCRIPT_DIR}/train_qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low.sh"
/home/vinhld8/Desktop/code/new_nothing/VLM_Distillation-main/project_commands.sh