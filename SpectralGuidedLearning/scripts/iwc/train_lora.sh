#!/usr/bin/env bash
# LoRA SFT arm -- any student (Qwen2.5-7B-Instruct, Qwen3-8B, ...), P-ALIGN data. Same recipe as the 1.5B full-FT arms
# (scripts/provenance/train_arm_r1-qwen-1.5b.sh): 3 epochs, eff. batch 32, lr cosine to MIN_LR,
# warmup 0.1, max_seq_len 32768, DeepSpeed ZeRO-2 offload -- except that only LoRA adapters are
# trained (r16 / alpha16 / dropout 0.05 on all attention + MLP projections, as scripts/iwc/train_iwc.sh).
# The output is an adapter-only checkpoint (--no-lora-merge); eval loads it through vLLM's LoRA support.
# Arms differ only in the per-token weights in <data-variant>.jsonl (loss_weights, applied by
# train_sft.py's MaskedSFTTrainer -- never train them through train_sft_unsloth.py, it drops them).
# Usage: MODEL_NAME=<hf id> TRACK=<track> scripts/iwc/train_lora.sh <arm-name> <data-variant> [train_sft.py args...]
#   e.g. MODEL_NAME=Qwen/Qwen3-8B TRACK=qwen3-8b-palign ... sft-nll-lora vanilla --objective nll
# Env: MODEL_NAME and TRACK (required), GPUS (default "0"), SEED, LR, MIN_LR, LORA_R, LORA_ALPHA, DS_CONFIG (empty = no DeepSpeed),
#      EFFECTIVE_BATCH (8 for qwen25-7b* tracks, else 32), MODEL_DTYPE (bfloat16 default | float32: fp32 weights + bf16 autocast; DeepSpeed is then off by default because
#      its bf16 engine would cast the module back to bf16).
set -euo pipefail
TRACK="${TRACK:?set TRACK, e.g. qwen25-7b-palign}"

ARM="${1:?usage: $0 <arm-name> <data-variant> [train_sft.py args...]}"
VARIANT="${2:?usage: $0 <arm-name> <data-variant> [train_sft.py args...]}"
shift 2
EXTRA_ARGS=("$@")

read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# ZeRO-2 offload JIT-compiles cpu_adam against system nvcc, which can trail the torch cuXXX build.
export DS_SKIP_CUDA_CHECK=1

# Single-node run: drop the PyTorchJob rendezvous env that would make torchrun hang.
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

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. Qwen/Qwen2.5-7B-Instruct}"
DATA_PATH="${BASE_PATH}/data/${TRACK}/train-${VARIANT}.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${ARM}-${TRACK}"
EPOCHS=3
# 5e-5 is what every existing LoRA script in this repo uses (spectral_lora_*, train_iwc.sh); LoRA
# often prefers a larger lr -- override with LR=1e-4 if the adapter barely moves.
LR="${LR:-5.0e-5}"
MIN_LR="${MIN_LR:-1.0e-5}"
WARMUP_RATIO=0.1
BATCH_SIZE=1
# Qwen2.5-7B trains with effective batch 8; every other student keeps 32 (override: EFFECTIVE_BATCH=<n>).
case "${TRACK}" in qwen25-7b*) _DEFAULT_EB=8 ;; *) _DEFAULT_EB=32 ;; esac
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-${_DEFAULT_EB}}"
(( EFFECTIVE_BATCH % GPUS_PER_NODE == 0 )) || { echo "GPU count ${GPUS_PER_NODE} must divide ${EFFECTIVE_BATCH}" >&2; exit 2; }
GRAD_ACC=$((EFFECTIVE_BATCH / (BATCH_SIZE * GPUS_PER_NODE)))   # bs1 x ga x n GPU = effective batch ${EFFECTIVE_BATCH}
ATTN=sdpa
LOG_INTERVAL=5
SEED="${SEED:-42}"
SAVE_STRATEGY=epoch
SAVE_TOTAL_LIMIT=2
MAX_SEQ_LEN=32768
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
case "${MODEL_DTYPE}" in bfloat16|float32) ;; *) echo "MODEL_DTYPE must be bfloat16 or float32" >&2; exit 2 ;; esac
if [[ "${MODEL_DTYPE}" == float32 ]]; then DEFAULT_DS=""; else DEFAULT_DS="${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"; fi
DS_CONFIG="${DS_CONFIG-${DEFAULT_DS}}"

[[ -f "${DATA_PATH}" ]] || { echo "missing ${DATA_PATH} -- run project_commands_lora_palign.sh (data phase) first" >&2; exit 1; }

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
OPTS+=" --model-dtype ${MODEL_DTYPE}"
OPTS+=" --use-lora"
OPTS+=" --lora-r ${LORA_R}"
OPTS+=" --lora-alpha ${LORA_ALPHA}"
OPTS+=" --lora-dropout ${LORA_DROPOUT}"
OPTS+=" --lora-target-modules ${LORA_TARGET_MODULES}"
OPTS+=" --no-lora-merge"
if [[ -n "${DS_CONFIG:-}" ]]; then
  OPTS+=" --deepspeed-config ${DS_CONFIG}"
fi

CMD="torchrun ${DISTRIBUTED_ARGS} -m sgl.training.train ${OPTS} ${EXTRA_ARGS[*]}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${ARM}-${TRACK}.log"
