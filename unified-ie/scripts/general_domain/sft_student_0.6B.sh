#! /bin/bash
# SFT of the Qwen3-0.6B student (full fine-tuning, CE only), from open-ed scripts/qwen/sft/sft_qwen3_0.6B.sh
# usage: bash scripts/general_domain/sft_student_0.6B.sh <data folder> [balance alpha]
#   bash scripts/general_domain/sft_student_0.6B.sh ee_v1          # every row once per epoch
#   bash scripts/general_domain/sft_student_0.6B.sh ee_v1 0.5      # square-root balancing (BalancedSampler)
# A balanced epoch is shorter (alpha 0.5 on ee_v*: 14.7k of 37.6k rows), so the default epoch count
# goes up to keep the number of updates about the same: 13 x 14.7k ~ 5 x 37.6k. EPOCHS=... overrides.

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

# DATA: folder under data/general_domain (ee_v1, ee_v2, ee_arg_v1, ee_arg_v2)
DATA=${1:-ee_v1}
ALPHA=${2:-}
# model
BASE_PATH=.
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
[ -d models/Qwen3-0.6B ] && CKPT="models/Qwen3-0.6B"   # local copy from download.txt
# data
DATA_DIR="${BASE_PATH}/processed_data/general_domain/${DATA}/qwen/"
# hp
BATCH_SIZE=4
LR=0.00005
GRAD_ACC=4
EVAL_BATCH_SIZE=16
if [ -n "${ALPHA}" ]; then
    EPOCHS=${EPOCHS:-13}
    RUN_NAME="sft_0.6B_${DATA}_bal${ALPHA}"
else
    EPOCHS=${EPOCHS:-5}
    RUN_NAME="sft_0.6B_${DATA}"
fi
# length
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/general_domain/${RUN_NAME}"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# score each source dataset in its own labels; dev answers every epoch, test answers when dev improves
OPTS+=" --label-map ${BASE_PATH}/data/general_domain/ee_label_map.tsv"
OPTS+=" --eval-gen-mode best"
# pad each batch to its longest row
OPTS+=" --dynamic-pad"
if [ -n "${ALPHA}" ]; then
    OPTS+=" --balance-alpha ${ALPHA}"
fi
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --warmup-ratio 0.1"
OPTS+=" --lr-decay-style wrmup_cosine"
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
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type lm"
# gen: greedy (no --do-sample)


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} ${@:3}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
