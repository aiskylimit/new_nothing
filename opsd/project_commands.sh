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

# Download HuggingFaceH4/aime_2024 to RAW_DATA_ROOT/eval/aime24 first (see download.txt).
if [[ ! -d "${PREPARED_DATA_ROOT}/eval/aime24" ]]; then
    python "${PROJECT_ROOT}/data/prepare_data.py" \
        --raw_root "${RAW_DATA_ROOT}" \
        --output_root "${PREPARED_DATA_ROOT}" \
        --only_eval aime24
fi

# Preserve completed evaluations for other benchmarks and any existing AIME24 results.
export OVERWRITE_EVAL=0

# Qwen3-4B and Qwen3-8B: train OPSD, then evaluate configured checkpoints
# on AIME24, AIME25, AIME26, and HMMT25.
for model in 4b 8b; do
    EVAL_DATASETS="aime24" bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" "${model}" base
    bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd "${model}"
    bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" "${model}" opsd
done

# Olmo-3-7B-Think was already trained: evaluate base and OPSD on AIME24 only.
EVAL_DATASETS="aime24" bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" olmo7b base
EVAL_DATASETS="aime24" bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" olmo7b opsd
