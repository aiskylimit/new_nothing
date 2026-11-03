#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

DOWNLOAD_ROOT=/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main
DOWNLOAD_DATA_DIR="${DOWNLOAD_ROOT}/train_data"
DATA_DIR="${PROJECT_DIR}/train_data"

TRAIN_ENV=/mnt/local/uvenvs/vlm-distill-train-3/bin/activate
TRAIN_DIR="${PROJECT_DIR}/script_train/qwen2_teacher_7b_fastvlm_student_05b"
EVAL_SCRIPT="${PROJECT_DIR}/project_commands_eval.sh"

BASE_MODEL="${BASE_MODEL:-${DOWNLOAD_ROOT}/models/KamilaMila/FastVLM-0.5B}"

source "${TRAIN_ENV}"
cd "${PROJECT_DIR}"

# Fail before torchrun launches four ranks if CuPy discovers headers from a
# different CUDA major version (for example nvidia/cu13 with CUDA 12 NVRTC).
python - <<'PY'
import cupy as cp
from cuml.cluster import DBSCAN

probe = cp.arange(4, dtype=cp.float32)
assert float(probe.sum().item()) == 6.0
print(
    "cuML preflight OK:",
    f"CuPy {cp.__version__}, runtime {cp.cuda.runtime.runtimeGetVersion()},",
    f"device {cp.cuda.runtime.getDevice()}",
)
PY

# ============================================================
# 1. Prepare datasets
# ============================================================

mkdir -p "${DATA_DIR}/ocr_vqa"

if ! cmp -s \
    "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" \
    "${DATA_DIR}/llava_v1_5_mix665k.json"; then

    cp -f \
        "${DOWNLOAD_DATA_DIR}/llava_v1_5_mix665k.json" \
        "${DATA_DIR}/llava_v1_5_mix665k.json"
fi

if ! cmp -s \
    "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" \
    "${DATA_DIR}/ocr_vqa/dataset.json"; then

    cp -f \
        "${DOWNLOAD_DATA_DIR}/ocr_vqa/dataset.json" \
        "${DATA_DIR}/ocr_vqa/dataset.json"
fi

unzip -q -o "${DOWNLOAD_DATA_DIR}/coco/train2017.zip" -d "${DATA_DIR}/coco"
unzip -q -o "${DOWNLOAD_DATA_DIR}/gqa/images.zip" -d "${DATA_DIR}/gqa"
unzip -q -o "${DOWNLOAD_DATA_DIR}/textvqa/train_val_images.zip" -d "${DATA_DIR}/textvqa"
unzip -q -o "${DOWNLOAD_DATA_DIR}/ocr_vqa/ocr_vqa_images.zip" -d "${DATA_DIR}/ocr_vqa"
unzip -q -o "${DOWNLOAD_DATA_DIR}/vg/images.zip" -d "${DATA_DIR}/vg"
unzip -q -o "${DOWNLOAD_DATA_DIR}/vg/images2.zip" -d "${DATA_DIR}/vg"
unzip -q -o "${DOWNLOAD_ROOT}/VLMEvalKit.zip" -d "${DOWNLOAD_ROOT}"


# ============================================================
# 2. SCVA + SIC layer-range tuning jobs
# ============================================================

JOBS=(
#   "train_qwen2_teacher_7b_fastvlm_student_05b_emkd.sh|qwen2_teacher_7b_fastvlm_student_05b_emkd"
#   "train_qwen2_teacher_7b_fastvlm_student_05b_mcw_kd.sh|qwen2_teacher_7b_fastvlm_student_05b_mcw_kd"
#   "train_qwen2_teacher_7b_fastvlm_student_05b_sre.sh|qwen2_teacher_7b_fastvlm_student_05b_sre"
#   "train_qwen2_teacher_7b_fastvlm_student_05b_scva.sh|qwen2_teacher_7b_fastvlm_student_05b_scva"
#   "train_qwen2_teacher_7b_fastvlm_student_05b_sic.sh|qwen2_teacher_7b_fastvlm_student_05b_sic"
#   "train_qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low.sh|qwen2_teacher_7b_fastvlm_student_05b_scva_sic_low"
  "train_qwen2_teacher_7b_fastvlm_student_05b_scva_sic_mid.sh|qwen2_teacher_7b_fastvlm_student_05b_scva_sic_mid"
  "train_qwen2_teacher_7b_fastvlm_student_05b_scva_sic_high.sh|qwen2_teacher_7b_fastvlm_student_05b_scva_sic_high"
)

for job in "${JOBS[@]}"; do

    IFS='|' read -r train_script run_name <<< "${job}"

    train_path="${TRAIN_DIR}/${train_script}"
    output_dir="${PROJECT_DIR}/outputs/${run_name}"

    if [[ ! -f "${train_path}" ]]; then
        echo "ERROR: Missing training script: ${train_path}" >&2
        exit 1
    fi

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    printf '\n============================================================\n'
    printf '[%s] START TRAIN: %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"
    printf 'GPUs: 4,5,6,7\n'
    printf '============================================================\n'

    if CUDA_VISIBLE_DEVICES=4,5,6,7 bash "${train_path}"; then

        printf '\n=== [%s] TRAIN EXIT SUCCESS: %s (exit_code=0) ===\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"

    else

        exit_code=$?

        printf '\n============================================================\n' >&2
        printf '[%s] TRAIN FAILED: %s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}" >&2
        printf 'EXIT CODE: %s\n' "${exit_code}" >&2
        printf 'TRAIN SCRIPT: %s\n' "${train_path}" >&2
        printf '============================================================\n' >&2

        exit "${exit_code}"

    fi

    # --------------------------------------------------------
    # EVAL
    # --------------------------------------------------------

    printf '\n============================================================\n'
    printf '[%s] START EVAL: %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"
    printf 'GPU: 4\n'
    printf '============================================================\n'

    if CUDA_VISIBLE_DEVICES=4 bash "${EVAL_SCRIPT}" \
        "${output_dir}" "${BASE_MODEL}"; then

        printf '\n=== [%s] EVAL SUCCESS: %s (exit_code=0) ===\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"

    else

        exit_code=$?

        printf '\n============================================================\n' >&2
        printf '[%s] EVAL FAILED: %s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}" >&2
        printf 'EXIT CODE: %s\n' "${exit_code}" >&2
        printf '============================================================\n' >&2

        exit "${exit_code}"

    fi

    printf '\n=== [%s] COMPLETED TRAIN + EVAL: %s ===\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${run_name}"

done

printf '\n============================================================\n'
printf '[%s] ALL TRAIN/EVAL JOBS COMPLETED SUCCESSFULLY\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')"
printf '============================================================\n'
