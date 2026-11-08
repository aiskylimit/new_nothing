#!/bin/bash

# Usage: bash runs/gpt2/run_t0.sh [BASE_PATH]
set -e -o pipefail
# Give background jobs process groups so cleanup also stops their workers.
set -m

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"

# Offline copy: environment and checkpoints are provisioned separately.

# Single physical GPU only. Ignore old per-stage multi-GPU overrides.
export CUDA_VISIBLE_DEVICES=6
GENERATE_GPU=6
OFFLINE_FIELD_GPU=6
DISTILLM2_FIELD_GPU=6
DISTILLM_GPU=6
FDD_GPU=6
DISTILLM2_GPU=6
CSD_GPU=6
BASE_PORT=${BASE_PORT:-20120}
DATA_PROCESS_WORKERS=${DATA_PROCESS_WORKERS:-4}

VELOCITY_PATH="${BASE_PATH}/results/gpt2/train/velocity_field_t0"
BASE_DIR=${BASE_DIR:-"/mnt/local/${PROJECT_NAME:-aiskylimit_new_nothing}/contra-kd"}
export TEACHER_PATH=${TEACHER_PATH:-"${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge"}
export STUDENT_PATH=${STUDENT_PATH:-"${BASE_DIR}/results/gpt2/train/init/gpt2-base"}
export RAW_DATA_ROOT=${RAW_DATA_ROOT:-"${BASE_DIR}/data/raw"}
export PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-"${BASE_DIR}/data/prepared"}
export PROCESSED_DATA_ROOT=${PROCESSED_DATA_ROOT:-"${BASE_DIR}/processed_data"}
export OPENWEBTEXT_DATASET_PATH=${OPENWEBTEXT_DATASET_PATH:-"${RAW_DATA_ROOT}/openwebtext-100k"}
export OPENWEBTEXT_TEXT_PATH=${OPENWEBTEXT_TEXT_PATH:-"${PREPARED_DATA_ROOT}/openwebtext/data.txt"}
export DOLLY_PROMPT_DATA_DIR=${DOLLY_PROMPT_DATA_DIR:-"${PREPARED_DATA_ROOT}/dolly"}
export DISTILLM2_DATA_DIR=${DISTILLM2_DATA_DIR:-"${BASE_DIR}/data/generated/distillm2/gpt2/t0"}
LOG_DIR=${LOG_DIR:-"${BASE_PATH}/logs/gpt2_t0"}
export PYTHONPATH=${BASE_PATH}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PIDS=()
JOB_NAMES=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill -- "-${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_jobs() {
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "Completed ${JOB_NAMES[$i]}."
        else
            job_status=$?
            echo "Failed ${JOB_NAMES[$i]} (exit ${job_status}). See ${LOG_DIR}." >&2
            return "${job_status}"
        fi
    done
    PIDS=()
    JOB_NAMES=()
}

mkdir -p "${LOG_DIR}"

echo "Step 1/6: Using the preinstalled contra environment (no installation or network access)."
echo "Step 2/6: Checking predownloaded teacher SFT and student initialization checkpoints..."
for checkpoint in "${STUDENT_PATH}" "${TEACHER_PATH}"
do
    if [ ! -f "${checkpoint}/config.json" ]; then
        echo "Missing downloaded checkpoint: ${checkpoint}/config.json" >&2
        exit 1
    fi
done

# All raw inputs must come from the snapshots listed in download.txt.
for input in dolly/raw.jsonl dolly/valid.jsonl self-inst/valid.jsonl \
    vicuna/valid.jsonl sinst/11_/valid.jsonl
do
    if [ ! -s "${RAW_DATA_ROOT}/${input}" ]; then
        echo "Missing downloaded input: ${RAW_DATA_ROOT}/${input}. Provision download.txt first." >&2
        exit 1
    fi
done

echo "Step 3/6: Preprocessing OpenWebText and full Dolly once..."
CUDA_VISIBLE_DEVICES="" python3 -u "${BASE_PATH}/tools/prepare_offline_dolly.py" \
    --raw-path "${RAW_DATA_ROOT}/dolly/raw.jsonl" \
    --output-dir "${DOLLY_PROMPT_DATA_DIR}" --dev-num 1000 \
    2>&1 | tee "${LOG_DIR}/prepare_dolly.log"
if [ ! -s "${OPENWEBTEXT_TEXT_PATH}" ]; then
    CUDA_VISIBLE_DEVICES="" python3 -u "${BASE_PATH}/tools/get_openwebtext.py" \
        2>&1 | tee "${LOG_DIR}/prepare_openwebtext.log"
fi

# Call the Python preprocessors directly, using the local tokenizer in every worker.
# Only the full Dolly dataset is needed; skip the prompt-only MiniLLM preprocessing.
CUDA_VISIBLE_DEVICES="" python3 -u "${BASE_PATH}/tools/process_data_pretrain.py" \
    --data-dir "$(dirname "${OPENWEBTEXT_TEXT_PATH}")" \
    --processed-data-dir "${PROCESSED_DATA_ROOT}/openwebtext/gpt2/512" \
    --model-path "${STUDENT_PATH}" \
    --model-type gpt2 \
    --max-length 512 \
    --train-num 22870 \
    --data-process-workers "${DATA_PROCESS_WORKERS}" \
    --dev-num 1000 \
    2>&1 | tee "${LOG_DIR}/preprocess_openwebtext.log"

CUDA_VISIBLE_DEVICES="" python3 -u "${BASE_PATH}/tools/process_data_dolly.py" \
    --data-dir "${RAW_DATA_ROOT}/dolly" \
    --processed-data-dir "${PROCESSED_DATA_ROOT}/dolly/full" \
    --model-path "${STUDENT_PATH}" \
    --model-type gpt2 \
    --max-prompt-length 256 \
    --data-process-workers "${DATA_PROCESS_WORKERS}" \
    --dev-num 1000 \
    2>&1 | tee "${LOG_DIR}/preprocess_dolly.log"

echo "Step 4/6: Generating DistiLLM2 response pairs on GPU ${GENERATE_GPU}..."
CUDA_VISIBLE_DEVICES=${GENERATE_GPU} bash "${BASE_PATH}/scripts/gpt2/distillm2/generate_data.sh" \
    "${BASE_PATH}" "${DISTILLM2_DATA_DIR}" "${TEACHER_PATH}" "${STUDENT_PATH}" \
    2>&1 | tee "${LOG_DIR}/generate_distillm2.log"

echo "Step 5/6: Pretraining two velocity fields sequentially on GPU 6..."
CUDA_VISIBLE_DEVICES=${OFFLINE_FIELD_GPU} bash "${BASE_PATH}/scripts/gpt2/train_velocity_field.sh" \
    "${BASE_PATH}" "$((BASE_PORT))" 1 \
    --model-path "${STUDENT_PATH}" \
    --teacher-model-path "${TEACHER_PATH}" \
    --save "${VELOCITY_PATH}" \
    > "${LOG_DIR}/field_offline.log" 2>&1 &
PIDS+=("$!")
JOB_NAMES+=("shared offline-Dolly field on GPU ${OFFLINE_FIELD_GPU}")
wait_for_jobs

CUDA_VISIBLE_DEVICES=${DISTILLM2_FIELD_GPU} bash "${BASE_PATH}/scripts/gpt2/distillm2/train_velocity_field_distillm2.sh" \
    "${BASE_PATH}" "$((BASE_PORT + 1))" 1 \
    --model-path "${STUDENT_PATH}" \
    --teacher-model-path "${TEACHER_PATH}" \
    --data-dir "${DISTILLM2_DATA_DIR}/formatted" \
    --save "${VELOCITY_PATH}/distillm2" \
    > "${LOG_DIR}/field_distillm2.log" 2>&1 &
PIDS+=("$!")
JOB_NAMES+=("DistiLLM2 field on GPU ${DISTILLM2_FIELD_GPU}")
wait_for_jobs

echo "Step 6/6: Training and evaluating four methods sequentially on GPU 6..."
CUDA_VISIBLE_DEVICES=${DISTILLM_GPU} EVAL_PORT=$((BASE_PORT + 20)) \
    bash "${BASE_PATH}/runs/gpt2/distillm_contra.sh" "${BASE_PATH}" "$((BASE_PORT + 10))" \
    > "${LOG_DIR}/distillm.log" 2>&1 &
PIDS+=("$!")
JOB_NAMES+=("DistiLLM +Contra t=0 on GPU ${DISTILLM_GPU}")
wait_for_jobs

CUDA_VISIBLE_DEVICES=${FDD_GPU} EVAL_PORT=$((BASE_PORT + 21)) \
    bash "${BASE_PATH}/runs/gpt2/fdd.sh" "${BASE_PATH}" "$((BASE_PORT + 11))" \
    > "${LOG_DIR}/fdd.log" 2>&1 &
PIDS+=("$!")
JOB_NAMES+=("FDD +Contra t=0 on GPU ${FDD_GPU}")
wait_for_jobs

CUDA_VISIBLE_DEVICES=${DISTILLM2_GPU} EVAL_PORT=$((BASE_PORT + 22)) \
    bash "${BASE_PATH}/runs/gpt2/distillm2_contra.sh" "${BASE_PATH}" "$((BASE_PORT + 12))" 1 \
    > "${LOG_DIR}/distillm2.log" 2>&1 &
PIDS+=("$!")
JOB_NAMES+=("DistiLLM2 +Contra t=0 on GPU ${DISTILLM2_GPU}")
wait_for_jobs

CUDA_VISIBLE_DEVICES=${CSD_GPU} EVAL_PORT=$((BASE_PORT + 23)) \
    bash "${BASE_PATH}/runs/csd/gpt2.sh" "${BASE_PATH}" "$((BASE_PORT + 13))" \
    > "${LOG_DIR}/csd.log" 2>&1 &
PIDS+=("$!")
JOB_NAMES+=("CSD +Contra t=0 on GPU ${CSD_GPU}")
wait_for_jobs

echo "All GPT-2 +Contra t=0 experiments completed. Logs: ${LOG_DIR}"
