#! /bin/bash

GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
# DATA: folder under data/general_domain (ee_v1, ee_v2, ee_arg_v1, ee_arg_v2)
DATA=${1:-ee_v1}
BASE_PATH=.
CKPT_NAME="qwen3-4B"
CKPT="Qwen/Qwen3-4B-Instruct-2507"
[ -d models/Qwen3-4B-Instruct-2507 ] && CKPT="models/Qwen3-4B-Instruct-2507"   # local copy from download.txt
# data
DATA_DIR="${BASE_PATH}/processed_data/general_domain/${DATA}/qwen/"
# hp
BATCH_SIZE=2
LR=0.0001
GRAD_ACC=8
EVAL_BATCH_SIZE=32
EPOCHS=3
# length
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/general_domain/sft_4B_${DATA}"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# score each source dataset in its own labels; dev answers every epoch, test answers when dev improves
OPTS+=" --label-map ${BASE_PATH}/data/general_domain/ee_label_map.tsv"
OPTS+=" --eval-gen-mode best"
# pad each batch to its longest row (the span loss in eventkd_0.6B_4B.sh keeps fixed padding)
OPTS+=" --dynamic-pad"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --warmup-ratio 0.1"
OPTS+=" --lr-decay-style wrmup_cosine"
# OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 460"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 32"
OPTS+=" --peft-lora-alpha 64"
OPTS+=" --peft-lora-dropout 0.1"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} ${@:2}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}

