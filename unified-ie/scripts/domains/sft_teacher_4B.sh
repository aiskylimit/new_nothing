#! /bin/bash
# LoRA SFT of the Qwen3-4B teacher on one source dataset of the biomedical or Vietnamese data, as the
# general_domain teachers (scripts/general_domain/ner/sft_teacher_4B_ner.sh: same hyperparameters, 5 epochs).
#   bash scripts/domains/sft_teacher_4B.sh <domain> <folder> <dataset>
#   specific_domain: ner bc5cdr, re ddi, ee phee
#   vietnamese:      ner vlsp2021, re vlsp2020, ee bkee
# Data: processed_data/<domain>/<folder>/qwen (specific_domain.zip / vietnamese.zip from download.txt) and
# data/<domain>/label_map.tsv. Saved under results/qwen3/<domain>/sft_4B_<folder>_<dataset>, where
# bam_0.6B_4B.sh (DOMAIN=<domain>) finds it. Standalone: run9.sh-style GPU and PORT (defaults below).

GPUS=(${GPU:-0})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${PORT:-29600}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE                   --nnodes $NNODES                   --node_rank $NODE_RANK                   --master_addr $MASTER_ADDR                   --master_port $MASTER_PORT"

DOMAIN=$1    # specific_domain | vietnamese
DATA=$2      # folder under processed_data/<domain>
DATASET=$3   # one source dataset of that folder: one teacher per dataset
if [ -z "${DOMAIN}" ] || [ -z "${DATA}" ] || [ -z "${DATASET}" ]; then
    echo "usage: bash scripts/domains/sft_teacher_4B.sh <domain> <folder> <dataset>"
    exit 1
fi
BASE_PATH=.
# model
CKPT_NAME="qwen3-4B"
CKPT="Qwen/Qwen3-4B-Instruct-2507"
[ -d models/Qwen3-4B-Instruct-2507 ] && CKPT="models/Qwen3-4B-Instruct-2507"   # local copy from download.txt
# data
DATA_DIR="${BASE_PATH}/processed_data/${DOMAIN}/${DATA}/qwen/"
[ -f ${DATA_DIR}/train_0.bin ] || { echo "missing ${DATA_DIR}: unpack ${DOMAIN}.zip (run9.sh does)"; exit 1; }
# hp: one 80GB GPU (B200): micro-batch 8 x GRAD_ACC 4 = effective batch 32, as the general_domain teachers
BATCH_SIZE=8
LR=0.0001
GRAD_ACC=4
EVAL_BATCH_SIZE=64
EPOCHS=${EPOCHS:-5}
# length: the rows were tokenized with --max-prompt-length 480 --max-length 768
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/${DOMAIN}/sft_4B_${DATA}_${DATASET}"
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
OPTS+=" --datasets ${DATASET}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# score in the dataset's own labels (NER and EE; RE needs no map); dev answers every epoch, test answers
# when the dev score improves
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
# length
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
