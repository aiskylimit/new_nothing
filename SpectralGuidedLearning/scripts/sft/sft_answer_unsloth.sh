#!/usr/bin/env bash
# Phase 5: answer-only SFT (no long CoT), Unsloth LoRA on one GPU. Hyperparameters match
# sft_unsloth_<track>.sh exactly so the only difference to the vanilla arm is the target text.
#   bash scripts/sft/sft_answer_unsloth.sh qwen25-7b
#   bash scripts/sft/sft_answer_unsloth.sh qwen3-8b
set -euo pipefail

TRACK="${1:?track is required: qwen25-7b or qwen3-8b}"
case "${TRACK}" in
  qwen25-7b) MODEL_DIR="Qwen2.5-7B-Instruct" ;;
  qwen3-8b) MODEL_DIR="Qwen3-8B" ;;
  *) echo "unknown track: ${TRACK}" >&2; exit 2 ;;
esac

read -ra GPUS <<< "${GPUS:-0}"
if [[ "${#GPUS[@]}" -ne 1 ]]; then
  echo "sft_answer_unsloth.sh: single-GPU only. Got GPUS='${GPUS[*]}'." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Unsloth needs its own venv (torch 2.9 / transformers 4.57 -- see spectral_guided_learning_train.txt);
# the main env (torch 2.13 / vllm) cannot hold it, so never fall back to scripts/setup.sh here.
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning_train}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: unsloth train env not found at ${PROJECT_ENV}; build it from spectral_guided_learning_train.txt or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/${MODEL_DIR}"
DATA_PATH="${BASE_PATH}/data/${TRACK}-answer/train-vanilla.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/answer-unsloth-${TRACK}"
EPOCHS=3
LR=5.0e-5
MIN_LR=1.0e-5
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=32
LOG_INTERVAL=5
SEED=42
SAVE_STRATEGY=epoch
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=6
OPTIM=adamw_torch
MAX_SEQ_LEN=32768
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

[[ -f "${DATA_PATH}" ]] || { echo "ERROR: ${DATA_PATH} missing; run scripts/data/data_answer.sh ${TRACK} first" >&2; exit 1; }

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
OPTS+=" --logging-steps ${LOG_INTERVAL}"
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-steps ${SAVE_STEPS}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --optim ${OPTIM}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"
OPTS+=" --use-lora"
OPTS+=" --lora-r ${LORA_R}"
OPTS+=" --lora-alpha ${LORA_ALPHA}"
OPTS+=" --lora-dropout ${LORA_DROPOUT}"
OPTS+=" --lora-target-modules ${LORA_TARGET_MODULES}"

CMD="python -m sgl.training.backends.unsloth ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/answer-unsloth-${TRACK}.log"
