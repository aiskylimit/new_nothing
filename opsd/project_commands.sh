#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /mnt/local/uvenvs/opsd/bin/activate

# Must match the destinations in download.txt.
PROJECT_NAME="aiskylimit_new_nothing"
BASE_DIR="/mnt/local/${PROJECT_NAME}/OPSD"
export MODEL_ROOT="${BASE_DIR}/models"
export RAW_DATA_ROOT="${BASE_DIR}/data/raw"
export PREPARED_DATA_ROOT="${BASE_DIR}/data/processed"
export OUTPUT_ROOT="${BASE_DIR}/outputs"
export RESULTS_ROOT="${BASE_DIR}/results/seed42_pass12"
export HF_HOME="${BASE_DIR}/.cache/huggingface"

# Two-GPU training and evaluation allocation.
export CUDA_VISIBLE_DEVICES="2,3"
export NUM_PROCESSES=2
export EVAL_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_MEMORY_UTILIZATION=0.6
export GPU_MEMORY_UTILIZATION=0.9
export MAIN_PROCESS_PORT=auto

# Prepare only datasets required by this run, without replacing existing data.
if [[ ! -d "${PREPARED_DATA_ROOT}/train" ]]; then
    python "${PROJECT_ROOT}/data/prepare_data.py" \
        --raw_root "${RAW_DATA_ROOT}" --output_root "${PREPARED_DATA_ROOT}" --only_train
fi
for dataset in aime25 aime26 hmmt25; do
    if [[ ! -d "${PREPARED_DATA_ROOT}/eval/${dataset}" ]]; then
        python "${PROJECT_ROOT}/data/prepare_data.py" \
            --raw_root "${RAW_DATA_ROOT}" --output_root "${PREPARED_DATA_ROOT}" --only_eval "${dataset}"
    fi
done

# Preserve completed evaluations from older runs without an explicit seed.
export EVAL_SEED=42
export EVAL_VAL_N=12
export OVERWRITE_EVAL=0
export EVAL_PASS_ONLY=1
export EVAL_DATASETS="aime25 aime26 hmmt25"

# Qwen3-8B: train OPSD, then evaluate pass@12 at the configured checkpoints.
bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd 8b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b opsd
