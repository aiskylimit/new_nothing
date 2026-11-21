#! /bin/bash
# SFT of the Qwen3-0.6B student (full fine-tuning, CE only) on all three tasks of the biomedical or Vietnamese
# data at once, as sft_student_0.6B_mt_v2.sh does for general_domain (same hyperparameters, 5 epochs).
# Baseline of the multi-teacher students (bam_0.6B_4B.sh with DOMAIN=<domain>, same data).
#   bash scripts/domains/sft_student_0.6B.sh specific_domain   # NER + RE + EE, 3 datasets, 13,270 train rows
#   bash scripts/domains/sft_student_0.6B.sh vietnamese        # NER + RE + EE, 3 datasets, 16,127 train rows
# Dev/test are scored per task; the checkpoint is picked on the mean of the task dev scores.
# Data: processed_data/<domain>/all/qwen and data/<domain>/label_map.tsv. Saved under
# results/qwen3/<domain>/sft_0.6B_all. Standalone: run9.sh-style GPU and PORT (defaults below).

GPUS=(${GPU:-0})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${PORT:-29600}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE                   --nnodes $NNODES                   --node_rank $NODE_RANK                   --master_addr $MASTER_ADDR                   --master_port $MASTER_PORT"

DOMAIN=$1   # specific_domain | vietnamese
if [ -z "${DOMAIN}" ]; then
    echo "usage: bash scripts/domains/sft_student_0.6B.sh <specific_domain|vietnamese>"
    exit 1
fi
DATA=all
BASE_PATH=.
# model
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
[ -d models/Qwen3-0.6B ] && CKPT="models/Qwen3-0.6B"   # local copy from download.txt
# data
DATA_DIR="${BASE_PATH}/processed_data/${DOMAIN}/${DATA}/qwen/"
[ -f ${DATA_DIR}/train_0.bin ] || { echo "missing ${DATA_DIR}: unpack ${DOMAIN}.zip (run9.sh does)"; exit 1; }
# hp: one 80GB GPU (B200): micro-batch 16 x GRAD_ACC 2 = effective batch 32, as the general_domain students
BATCH_SIZE=16
LR=0.00005
GRAD_ACC=2
EVAL_BATCH_SIZE=128
EPOCHS=5
# length
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/${DOMAIN}/sft_0.6B_${DATA}"
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
# score each source dataset in its own labels; dev answers every epoch, test answers when the mean of the
# task scores on dev improves
OPTS+=" --label-map ${BASE_PATH}/data/${DOMAIN}/label_map.tsv"
OPTS+=" --eval-gen-mode best"
# pad each batch to its longest row
OPTS+=" --dynamic-pad"
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
# length: the rows were tokenized with --max-prompt-length 480 --max-length 768
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 480"
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
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
