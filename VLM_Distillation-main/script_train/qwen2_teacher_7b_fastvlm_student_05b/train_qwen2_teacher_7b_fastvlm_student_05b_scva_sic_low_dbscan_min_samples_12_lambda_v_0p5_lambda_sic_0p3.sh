#!/usr/bin/env bash
# SCVA + SIC low weight ablation: min_samples=12, lambda_v=0.5, lambda_sic=0.3.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low_dbscan_min_samples_12_lambda_v_0p5_lambda_sic_0p3}"
export PERCENT_DATA="${PERCENT_DATA:-0.3}"
export SCVA_DBSCAN_MIN_SAMPLES="${SCVA_DBSCAN_MIN_SAMPLES:-12}"
export SCVA_SIC_LAMBDA_V="${SCVA_SIC_LAMBDA_V:-0.5}"
export SCVA_SIC_LAMBDA_SIC="${SCVA_SIC_LAMBDA_SIC:-0.3}"

exec bash "${SCRIPT_DIR}/train_qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low.sh"
