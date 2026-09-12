#!/usr/bin/env bash
# Phase 5: masked SFT -- spectral, DeepSeek-R1-Distill-Qwen-7B track.
# Unsloth FULL fine-tuning on a SINGLE GPU (no LoRA, no DeepSpeed, no torchrun).
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"   # Unsloth OSS is single-GPU; pin the first listed GPU
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-7B"
DATA_PATH="${BASE_PATH}/data/r1-qwen-7b/train-spectral.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/spectral-r1-qwen-7b"
EPOCHS=3
LR=1.0e-5              # full-FT-safe lr (LoRA used 5e-5); kept for comparability with prior full-FT
MIN_LR=1.0e-6
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=8            # bs1 x ga8 = effective batch 8 (single GPU)
OPTIM=adamw_8bit      # 8-bit Adam keeps 7B optimizer state ~14GB (fp32 Adam ~56GB) so full-FT fits 80GB
LOG_INTERVAL=5
SEED=42
SAVE_STRATEGY=epoch
SAVE_TOTAL_LIMIT=2
MAX_SEQ_LEN=32768

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --learning-rate ${LR}"
OPTS+=" --min-learning-rate ${MIN_LR}"
OPTS+=" --warmup-ratio ${WARMUP_RATIO}"
OPTS+=" --per-device-batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --optim ${OPTIM}"
OPTS+=" --logging-steps ${LOG_INTERVAL}"
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"

CMD="python ${BASE_PATH}/src/train_sft_unsloth.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/spectral-r1-qwen-7b.log"
