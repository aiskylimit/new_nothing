#!/usr/bin/env bash
# Provenance-aware SFT arms -- DeepSeek-R1-Distill-Qwen-1.5B, P-ALIGN data, FULL fine-tuning with
# exactly the IWC / spectral-full recipe (scripts/iwc/train_iwc_r1-qwen-1.5b.sh): 3 epochs, eff.
# batch 32, lr 5e-5 cosine to 1e-5, warmup 0.1, max_seq_len 32768, DeepSpeed ZeRO-2 offload (fp32
# master weights). Arms differ only in the per-token objective passed through to train_sft.py.
# Usage: scripts/provenance/train_arm_r1-qwen-1.5b.sh <arm-name> <data-variant> [train_sft.py args...]
#   e.g. ... prov-nll-dft provenance --prefix-objective nll --cont-objective dft
set -euo pipefail
# Data/checkpoint namespace: r1-qwen-1.5b = s1K-1.1 track; e.g. r1-qwen-1.5b-palign = P-ALIGN data.
TRACK="${TRACK:-r1-qwen-1.5b-palign}"

ARM="${1:?usage: $0 <arm-name> <data-variant> [train_sft.py args...]}"
VARIANT="${2:?usage: $0 <arm-name> <data-variant> [train_sft.py args...]}"
shift 2
EXTRA_ARGS=("$@")

read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# ZeRO-2 offload (only if DS_CONFIG is set) JIT-compiles cpu_adam against system nvcc, which can
# trail the torch cuXXX build -- skip that version check.
export DS_SKIP_CUDA_CHECK=1

# The cluster (PyTorchJob pod) injects PET_RDZV_BACKEND=c10d / PET_RDZV_ENDPOINT=<worker-0>:23456 /
# TORCHELASTIC_*; torchrun reads those over --master_addr and hangs in "Rendezvous'ing worker group"
# waiting on that endpoint. This is a single-node run: drop them and pin the static backend.
for _v in $(compgen -e PET_) $(compgen -e TORCHELASTIC_); do unset "$_v"; done
MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --rdzv_backend static \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE_PATH}/.." && pwd)/iwc}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
DATA_PATH="${BASE_PATH}/data/${TRACK}/train-${VARIANT}.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${ARM}-${TRACK}"
EPOCHS=3
LR="${LR:-5.0e-5}"
MIN_LR="${MIN_LR:-1.0e-5}"
WARMUP_RATIO=0.1
BATCH_SIZE=1
EFFECTIVE_BATCH=32
(( EFFECTIVE_BATCH % GPUS_PER_NODE == 0 )) || { echo "GPU count ${GPUS_PER_NODE} must divide ${EFFECTIVE_BATCH}" >&2; exit 2; }
GRAD_ACC=$((EFFECTIVE_BATCH / (BATCH_SIZE * GPUS_PER_NODE)))   # bs1 x ga x n GPU = effective batch 32
ATTN=sdpa             # train_sft.py uses the Trainer default optimizer (adamw_torch), as before
LOG_INTERVAL=5
SEED="${SEED:-42}"
SAVE_STRATEGY=epoch
SAVE_TOTAL_LIMIT=2
MAX_SEQ_LEN=32768
DS_CONFIG="${DS_CONFIG-${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json}"

[[ -f "${DATA_PATH}" ]] || { echo "missing ${DATA_PATH} -- run scripts/data/data_r1-qwen-1.5b.sh / src/sgl/transforms/provenance.py first" >&2; exit 1; }

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
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"
OPTS+=" --no-use-lora"
if [[ -n "${DS_CONFIG:-}" ]]; then
  OPTS+=" --deepspeed-config ${DS_CONFIG}"
fi

CMD="torchrun ${DISTRIBUTED_ARGS} -m sgl.training.train ${OPTS} ${EXTRA_ARGS[*]}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${ARM}-${TRACK}.log"
