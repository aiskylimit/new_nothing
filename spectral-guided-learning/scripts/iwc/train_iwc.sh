#!/usr/bin/env bash
# Phase 5: train an IWC arm. Hyperparameters match scripts/spectral exactly.
# Usage: scripts/iwc/train_iwc.sh {qwen25-7b|qwen3-8b} {iwc|iwc-stable}
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

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes 1 --node_rank 0 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/${MODEL_DIR}"
DATA_PATH="${BASE_PATH}/data/${TRACK}/train-${VARIANT}.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${VARIANT}-${TRACK}"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"

OPTS="--model-name ${MODEL_NAME} --data-path ${DATA_PATH} --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs 3 --learning-rate 5.0e-5 --min-learning-rate 1.0e-5 --warmup-ratio 0.1"
OPTS+=" --per-device-batch-size 1 --gradient-accumulation-steps 16 --attn-implementation sdpa"
OPTS+=" --logging-steps 5 --save-strategy epoch --save-steps 500 --save-total-limit 6 --seed 42"
OPTS+=" --use-lora --lora-r 16 --lora-alpha ${LORA_ALPHA:-16} --lora-dropout 0.05"
OPTS+=" --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --no-lora-merge"
OPTS+=" --deepspeed-config ${DS_CONFIG} --max-seq-len 32768"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${VARIANT}-${TRACK}.log"
