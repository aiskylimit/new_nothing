#!/usr/bin/env bash
set -euo pipefail

TRACK="${1:?track is required: qwen25-7b or qwen3-8b}"
VARIANT="${2:?variant is required: iwc or iwc-stable}"
case "${TRACK}" in
  qwen25-7b) MODEL_DIR="Qwen2.5-7B-Instruct" ;;
  qwen3-8b) MODEL_DIR="Qwen3-8B" ;;
  *) echo "unknown track: ${TRACK}" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  iwc|iwc-stable) ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

read -ra GPUS <<< "${GPUS:-0}"
if [[ "${#GPUS[@]}" -ne 1 ]]; then
  echo "train_iwc_unsloth.sh: single-GPU only. Got GPUS='${GPUS[*]}'." >&2
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
DATA_PATH="${BASE_PATH}/data/${TRACK}/train-${VARIANT}.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${VARIANT}-unsloth-${TRACK}"
OPTIM=adamw_torch

OPTS="--model-name ${MODEL_NAME} --data-path ${DATA_PATH} --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs 3 --learning-rate 5.0e-5 --min-learning-rate 1.0e-5 --warmup-ratio 0.1"
OPTS+=" --per-device-batch-size 1 --gradient-accumulation-steps 32"
OPTS+=" --logging-steps 5 --save-strategy epoch --save-steps 500 --save-total-limit 6 --seed 42"
OPTS+=" --optim ${OPTIM} --max-seq-len 32768"
OPTS+=" --use-lora --lora-r 16 --lora-alpha 16 --lora-dropout 0.05"
OPTS+=" --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

CMD="python -m sgl.training.backends.unsloth ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${VARIANT}-unsloth-${TRACK}.log"
