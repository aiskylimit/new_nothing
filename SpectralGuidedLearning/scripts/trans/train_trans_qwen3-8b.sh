#!/usr/bin/env bash
# Phase 5: masked SFT + L_trans (next-step representation prediction), Qwen3-8B track.
# Hyperparameters are scripts/sft/sft_qwen3-8b.sh's exactly (LoRA r16/alpha32, 3 epochs, lr 5e-5,
# effective batch 32); only the objective differs, so the run is comparable to the SFT baseline.
#
#   bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.3            # config 4: SFT + L_trans, lambda 0.3
#   bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.1            # config 3
#   bash scripts/trans/train_trans_qwen3-8b.sh spectral 0.3           # config 5: SGL + L_trans
#   bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.3 shuffle    # config 6: shuffled-target control
#
# GPUS="0 1" (default) runs DDP over two GPUs with grad-accum 16; GPUS="0" uses grad-accum 32,
# so the effective batch is 32 either way. Extra knobs via env: TRANS_LAYER, TRANS_LR, TRANS_HIDDEN.
set -euo pipefail

VARIANT="${1:-vanilla}"
LAMBDA="${2:-0.3}"
CONTROL="${3:-}"
case "${VARIANT}" in
  vanilla|spectral|iwc-stable) ;;
  *) echo "unknown mask variant: ${VARIANT} (vanilla|spectral|iwc-stable)" >&2; exit 2 ;;
esac

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# ZeRO-2 offload JIT-compiles cpu_adam against system nvcc, which can trail the torch cuXXX
# build (see docs/server-runbook.md CUDAMismatchException) -- skip that version check.
export DS_SKIP_CUDA_CHECK=1

# The cluster (PyTorchJob pod) injects PET_RDZV_BACKEND=c10d / PET_RDZV_ENDPOINT=<worker-0>:23456 /
# TORCHELASTIC_*; torchrun reads those over --master_addr and hangs in "Rendezvous'ing worker group"
# waiting on that endpoint. This is a single-node run: drop them and pin the static backend.
for _v in $(compgen -e PET_) $(compgen -e TORCHELASTIC_); do unset "$_v"; done
MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --rdzv_backend static --nnodes 1 --node_rank 0 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-8B"
DATA_PATH="${BASE_PATH}/data/qwen3-8b/train-${VARIANT}-trans.jsonl"
TAG="trans-${VARIANT}-l${LAMBDA}${CONTROL:+-${CONTROL}}-qwen3-8b"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${TAG}"
EPOCHS=3
LR=5.0e-5
MIN_LR=1.0e-5
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=$(( 32 / GPUS_PER_NODE ))   # bs1 x ga x n_gpu = effective batch 32
ATTN=sdpa
LOG_INTERVAL=5
SEED=42
SAVE_STRATEGY=epoch
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=6
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"
MAX_SEQ_LEN=32768
# L_trans: layer 24 of 36 (2/3 depth), 8.4M-param predictor with its own lr, lambda ramped over
# the LR warmup, grad-norm ratio logged every 50 optimizer steps.
TRANS_LAYER="${TRANS_LAYER:-24}"
TRANS_LR="${TRANS_LR:-2e-4}"
TRANS_HIDDEN="${TRANS_HIDDEN:-1024}"
TRANS_GRAD_LOG_INTERVAL="${TRANS_GRAD_LOG_INTERVAL:-50}"

[[ -f "${DATA_PATH}" ]] || { echo "ERROR: ${DATA_PATH} missing -- run scripts/trans/build_trans_qwen3-8b.sh ${VARIANT}" >&2; exit 1; }

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
OPTS+=" --attn-implementation ${ATTN}"
OPTS+=" --logging-steps ${LOG_INTERVAL}"
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-steps ${SAVE_STEPS}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --use-lora"
OPTS+=" --lora-r ${LORA_R}"
OPTS+=" --lora-alpha ${LORA_ALPHA}"
OPTS+=" --lora-dropout ${LORA_DROPOUT}"
OPTS+=" --lora-target-modules ${LORA_TARGET_MODULES}"
OPTS+=" --no-lora-merge"
OPTS+=" --deepspeed-config ${DS_CONFIG}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"
OPTS+=" --metrics-log ${BASE_PATH}/logs/${TAG}-metrics.json"
OPTS+=" --trans-lambda ${LAMBDA}"
OPTS+=" --trans-layer ${TRANS_LAYER}"
OPTS+=" --trans-lr ${TRANS_LR}"
OPTS+=" --trans-hidden ${TRANS_HIDDEN}"
OPTS+=" --trans-grad-log-interval ${TRANS_GRAD_LOG_INTERVAL}"
[[ "${CONTROL}" == "shuffle" ]] && OPTS+=" --trans-shuffle-targets"
[[ "${RESUME:-0}" == "1" ]] && OPTS+=" --resume"

CMD="torchrun ${DISTRIBUTED_ARGS} -m sgl.training.train ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${TAG}.log"
