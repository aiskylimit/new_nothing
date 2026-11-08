#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Must match @PROJECT@ expanded by the provisioner in download.txt.
PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothing}"
BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/contra-kd}"
ENV_DIR="${ENV_DIR:-/mnt/local/uvenvs/contra}"
if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
    echo "Missing environment: ${ENV_DIR}. Provision contra.txt before running offline." >&2
    exit 1
fi
source "${ENV_DIR}/bin/activate"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export RAW_DATA_ROOT="${BASE_DIR}/data/raw"
export PREPARED_DATA_ROOT="${BASE_DIR}/data/prepared"
export PROCESSED_DATA_ROOT="${BASE_DIR}/processed_data"
export OPENWEBTEXT_DATASET_PATH="${RAW_DATA_ROOT}/openwebtext-100k"
export OPENWEBTEXT_TEXT_PATH="${PREPARED_DATA_ROOT}/openwebtext/data.txt"
export DOLLY_PROMPT_DATA_DIR="${PREPARED_DATA_ROOT}/dolly"
export DISTILLM2_DATA_DIR="${BASE_DIR}/data/generated/distillm2/gpt2/t0"
export TEACHER_PATH="${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}"
export STUDENT_PATH="${STUDENT_PATH:-${BASE_DIR}/results/gpt2/train/init/gpt2-base}"

# Every generation, field, training and evaluation job runs serially on GPU 6.
export CUDA_VISIBLE_DEVICES=6
export BASE_PORT="${BASE_PORT:-20120}"
export DATA_PROCESS_WORKERS="${DATA_PROCESS_WORKERS:-4}"

cd "${PROJECT_ROOT}"
exec bash "${PROJECT_ROOT}/runs/gpt2/run_t0.sh" "${PROJECT_ROOT}"
