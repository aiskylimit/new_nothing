#!/usr/bin/env bash
# SCVA + SIC low region ablation: min_samples=12, regions=2, intervention batch=2.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low_dbscan_min_samples_12_regions_2_intervention_batch_2}"
export SCVA_DBSCAN_MIN_SAMPLES="${SCVA_DBSCAN_MIN_SAMPLES:-12}"
export SCVA_SIC_LAMBDA_V="${SCVA_SIC_LAMBDA_V:-0.3}"
export SCVA_SIC_LAMBDA_SIC="${SCVA_SIC_LAMBDA_SIC:-0.3}"
export SIC_NUM_REGIONS="${SIC_NUM_REGIONS:-2}"
export SIC_INTERVENTION_BATCH_SIZE="${SIC_INTERVENTION_BATCH_SIZE:-2}"

exec bash "${SCRIPT_DIR}/train_qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low.sh"
