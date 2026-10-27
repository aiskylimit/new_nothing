#!/bin/bash
# E-SCOT (paper 15849): student GPT2-1.5B, teacher Qwen2.5-7B-Instruct (+ LoRA Dolly SFT)
GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=52$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"

# model — student: model_hub/GPT2-1.5B | teacher: model_hub/Qwen2.5-7B-Instruct | teacher LoRA: lora_path/Qwen2.5-7B-Instruct
CKPT_TYPE="gpt2"
CKPT_NAME="GPT2-1.5B"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen2.5-7B-Instruct"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"
TEACHER_PEFT_PATH="${BASE_PATH}/lora_path/${TEACHER_MODEL_NAME}"

# data
DATA_DIR="${BASE_PATH}/data/dolly/"

# task — E-SCOT paper §4.1: 10 epochs, τ=2.0, ρ=0.5, λ=0.25
TASK="E_SCOT"
BATCH_SIZE=4
LR=0.001
GRAD_ACC=2
EVAL_BATCH_SIZE=8
EPOCH=${EPOCH:-15}
WARMUP_ITERS=0
KD_RATE=0.5
KD_TEMP=2.0
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1

# E-SCOT-specific hyperparams
E_SCOT_LAMBDA=0.25
E_SCOT_ALPHAS="0.5,1.0,2.0"

# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
CRITERION="E_SCOT"
KD_OBJ="forward_kl"

CONFIG="${KD_OBJ}-lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__escot^λ=${E_SCOT_LAMBDA}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
# seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"

# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# task
OPTS+=" --task ${TASK}"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters ${WARMUP_ITERS}"
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

# E-SCOT
OPTS+=" --e-scot-lambda ${E_SCOT_LAMBDA}"
OPTS+=" --e-scot-alphas ${E_SCOT_ALPHAS}"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"

OPTS+=" --precision ${PRECISION}"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"

# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"

${CMD} 2>&1 | tee "${SAVE_PATH}/train.log"
