#! /bin/bash
# LoRA SFT of the Qwen3-4B teacher on FewRel (sentence-level RE)
# Standalone: data and hyperparameters are fixed here; run.sh starts it with GPU and PORT set (defaults below).

GPUS=(${GPU:-5})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${PORT:-29506}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE                   --nnodes $NNODES                   --node_rank $NODE_RANK                   --master_addr $MASTER_ADDR                   --master_port $MASTER_PORT"

DATA=re_fewrel
BASE_PATH=.
# model
CKPT_NAME="qwen3-4B"
CKPT="Qwen/Qwen3-4B-Instruct-2507"
[ -d models/Qwen3-4B-Instruct-2507 ] && CKPT="models/Qwen3-4B-Instruct-2507"   # local copy from download.txt
# data
DATA_DIR="${BASE_PATH}/processed_data/general_domain/${DATA}/qwen/"
# hp: one 80GB GPU (B200): micro-batch 8 x GRAD_ACC 4 = effective batch 32, as 2 x 8 x 2 GPUs before
BATCH_SIZE=8
LR=0.0001
GRAD_ACC=4
EVAL_BATCH_SIZE=64
EPOCHS=${EPOCHS:-5}   # the first teachers (2026-10-26/27) had 3; run.sh trains those 2 more from their last checkpoint
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
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# dev answers every epoch, test answers when dev triple F1 improves
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
# PEFT_PATH=<teacher checkpoint, e.g. results/.../sft_4B_ee_v1_ace/e3-.../288> (EPOCHS=<n> for how many more):
# training goes on from that adapter (its own rank, alpha, target modules; fresh optimizer and lr schedule),
# saved next to the first run with the suffix -from<step> (passed as --opt=value: argparse takes a bare "-from..." for a flag)
[ -n "${PEFT_PATH:-}" ] && OPTS+=" --peft-path ${PEFT_PATH} --save-additional-suffix=-from$(basename ${PEFT_PATH})"
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
