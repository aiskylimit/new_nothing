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
export RESULTS_ROOT="${BASE_DIR}/results"
export HF_HOME="${BASE_DIR}/.cache/huggingface"

# Two-GPU training and evaluation allocation.
export CUDA_VISIBLE_DEVICES="2,3"
export NUM_PROCESSES=2
export EVAL_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_MEMORY_UTILIZATION=0.6
export GPU_MEMORY_UTILIZATION=0.9
export MAIN_PROCESS_PORT=auto

# Prepare downloaded datasets for training and evaluation.
python "${PROJECT_ROOT}/data/prepare_data.py" \
    --raw_root "${RAW_DATA_ROOT}" \
    --output_root "${PREPARED_DATA_ROOT}" \
    --overwrite

# Evaluate each base model once, independently of method training and checkpoint evaluation.
# bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd 4b
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b opsd

# bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd 8b
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b opsd

# bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd olmo7b  # completed
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" olmo7b opsd  # completed

# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b base  # completed
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b base  # completed
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" olmo7b base

# bash "${PROJECT_ROOT}/scripts/run_training.sh" grpo 4b
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b grpo

# bash "${PROJECT_ROOT}/scripts/run_training.sh" grpo 8b
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b grpo

# bash "${PROJECT_ROOT}/scripts/run_training.sh" sft 4b
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b sft

# bash "${PROJECT_ROOT}/scripts/run_training.sh" sft 8b
# bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b sft

bash "${PROJECT_ROOT}/scripts/run_training.sh" grpo olmo7b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" olmo7b grpo

bash "${PROJECT_ROOT}/scripts/run_training.sh" sft olmo7b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" olmo7b sft
