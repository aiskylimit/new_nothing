#!/bin/bash
# GPU 0 — Idea 1: boundary-mode ablation on Qwen2.5-7B-Instruct → GPT2-1.5B
# Sequential: nearest → containing  (DSKDv2 + IMPACT, full training)
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.3}"
# Prefer active conda env on PATH when present.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/bin" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
else
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"

DSKD_ROOT="${BASE_PATH}/DSKDv2"
LOG_DIR="${BASE_PATH}/logs/ablation"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/gpu0_boundary_modes_summary.tsv"
echo -e "mode\tstart\tend\tstatus\tsave_path" > "${SUMMARY}"

CKPT_TYPE="gpt2"
CKPT_NAME="GPT2-1.5B"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen2.5-7B-Instruct"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"
TEACHER_PEFT_PATH="${BASE_PATH}/lora_path/${TEACHER_MODEL_NAME}"
DATA_DIR="${BASE_PATH}/data/dolly/"

BATCH_SIZE=4
LR=0.001
GRAD_ACC=2
EVAL_BATCH_SIZE=8
EPOCH=15
KD_RATE=0.8
KD_TEMP=2.0
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1
IMPACT_LAMBDA=0.25
IMPACT_TOP_K=4
IMPACT_BI_TAU=1.0
IMPACT_LAMBDA_REG=1.0
PROJECTOR_LR=0.0005
TOPK_VOCAB=-1
MAX_LENGTH=512
PRECISION="bf16"
CRITERION="dual_space_kd_v2_impact"
KD_OBJ="forward_kl"
SEED=10
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
MASTER_ADDR=localhost

run_boundary() {
  local MODE="$1"
  local PORT="61$((RANDOM % 90 + 10))"
  local TASK="dskd_v2_impact_boundary_${MODE}"
  local CONFIG="${KD_OBJ}-lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
  local SETTING="criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__impact^λ=${IMPACT_LAMBDA}__impact^k=${IMPACT_TOP_K}__boundary=${MODE}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x1=$((BATCH_SIZE * GRAD_ACC))__lr=${LR}__proj^lr=${PROJECTOR_LR}"
  local SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
  mkdir -p "${SAVE_PATH}"

  local START END STATUS
  START="$(date -Iseconds)"
  echo "========== [GPU0] boundary_mode=${MODE} START ${START} =========="

  DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${PORT}"

  OPTS=""
  OPTS+=" --base-path ${DSKD_ROOT}"
  OPTS+=" --model-type ${CKPT_TYPE}"
  OPTS+=" --model-path ${CKPT_PATH}"
  OPTS+=" --model-dtype ${PRECISION}"
  OPTS+=" --n-gpu ${GPUS_PER_NODE}"
  OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
  OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
  OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
  OPTS+=" --teacher-model-fp16"
  OPTS+=" --gradient-checkpointing"
  OPTS+=" --data-dir ${DATA_DIR}"
  OPTS+=" --num-workers 0"
  OPTS+=" --dev-num 1000"
  OPTS+=" --task ${TASK}"
  OPTS+=" --lr ${LR}"
  OPTS+=" --batch-size ${BATCH_SIZE}"
  OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
  OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
  OPTS+=" --warmup-iters 0"
  OPTS+=" --lr-decay-style cosine"
  OPTS+=" --weight-decay 1e-2"
  OPTS+=" --clip-grad 1.0"
  OPTS+=" --num-epochs ${EPOCH}"
  OPTS+=" --kd-rate ${KD_RATE}"
  OPTS+=" --kd-temperature ${KD_TEMP}"
  OPTS+=" --kd-objective ${KD_OBJ}"
  OPTS+=" --peft lora"
  OPTS+=" --peft-lora-r ${LORA_RANK}"
  OPTS+=" --peft-lora-alpha ${LORA_ALPHA}"
  OPTS+=" --peft-lora-dropout ${LORA_DROPOUT}"
  OPTS+=" --init-t2s-projector"
  OPTS+=" --init-s2t-projector"
  OPTS+=" --projector-lr ${PROJECTOR_LR}"
  OPTS+=" --topk-vocab ${TOPK_VOCAB}"
  OPTS+=" --impact-lambda ${IMPACT_LAMBDA}"
  OPTS+=" --impact-top-k ${IMPACT_TOP_K}"
  OPTS+=" --impact-bi-tau ${IMPACT_BI_TAU}"
  OPTS+=" --impact-lambda-reg ${IMPACT_LAMBDA_REG}"
  OPTS+=" --impact-bi-calib-offline"
  OPTS+=" --impact-bi-calib-samples 256"
  OPTS+=" --impact-boundary-mode ${MODE}"
  OPTS+=" --max-length ${MAX_LENGTH}"
  OPTS+=" --max-prompt-length 256"
  OPTS+=" --do-train"
  OPTS+=" --do-valid"
  OPTS+=" --eval-gen"
  OPTS+=" --save-interval 1"
  OPTS+=" --eval-interval 1"
  OPTS+=" --log-interval 50"
  OPTS+=" --save-dir ${SAVE_PATH}"
  OPTS+=" --keep-best-n-checkpoints 1"
  OPTS+=" --criterion ${CRITERION}"
  OPTS+=" --seed ${SEED}"
  OPTS+=" --deepspeed"
  OPTS+=" --deepspeed_config ${DSKD_ROOT}/configs/deepspeed/ds_config_bf16.json"
  OPTS+=" --do-sample"
  OPTS+=" --top-k 0"
  OPTS+=" --top-p 1.0"
  OPTS+=" --temperature 1.0"

  export NCCL_DEBUG=""
  export WANDB_DISABLED=True
  export TF_CPP_MIN_LOG_LEVEL=3
  export PYTHONPATH="${DSKD_ROOT}"

  set +e
  torchrun ${DISTRIBUTED_ARGS} "${DSKD_ROOT}/code/distillation.py" ${OPTS} \
    2>&1 | tee "${SAVE_PATH}/train.log"
  local RC=${PIPESTATUS[0]}
  set -e

  END="$(date -Iseconds)"
  if [[ ${RC} -eq 0 ]]; then STATUS="ok"; else STATUS="fail_rc${RC}"; fi
  echo -e "${MODE}\t${START}\t${END}\t${STATUS}\t${SAVE_PATH}" >> "${SUMMARY}"
  echo "========== [GPU0] boundary_mode=${MODE} END ${END} status=${STATUS} =========="
  return ${RC}
}

echo "[GPU0] Waiting for model weights..."
for p in "${CKPT_PATH}/config.json" "${TEACHER_MODEL_PATH}/config.json" "${TEACHER_PEFT_PATH}/adapter_config.json"; do
  while [[ ! -f "${p}" ]]; do
    echo "  missing ${p}; sleep 30s"
    sleep 30
  done
done
echo "[GPU0] Models ready."

run_boundary nearest || true
run_boundary containing || true

echo "[GPU0] All boundary-mode runs finished. Summary: ${SUMMARY}"
cat "${SUMMARY}"
