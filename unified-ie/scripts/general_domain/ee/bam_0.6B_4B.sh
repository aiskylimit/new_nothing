#! /bin/bash
# Multi-teacher KD as in BAM (Clark et al. 2019): the Qwen3-0.6B student (full fine-tuning) learns event
# extraction (ee_v1 or ee_v2: ACE, MAVEN, RAMS, GENEVA) from four Qwen3-4B LoRA teachers, one per source
# dataset; each row is distilled from the teacher of its own dataset. Same data and hyperparameters as
# sft_student_0.6B_ee_v*.sh (effective batch 32, 5 epochs), so the SFT student is the baseline.
#   bash scripts/general_domain/ee/bam_0.6B_4B.sh ee_v1
# Each part is a switch (env variable, default = BAM); the run dir is named after them:
#   KD_SCHEDULE=anneal    BAM teacher annealing (KD weight 1 -> 0); constant = fixed KD_RATIO
#   KD_TYPE=fkl           forward KL, BAM's loss; sfkl = skewed forward KL of EventKD
#   W_SPAN=0              span loss off; 1.0 = EventKD span loss on teacher layers 33/36
#   TEACHERS=             name=lora_dir,...; empty = best dev epoch of results/.../sft_4B_<data>_<dataset>
# e.g. KD_SCHEDULE=constant KD_TYPE=sfkl W_SPAN=1.0 bash scripts/general_domain/ee/bam_0.6B_4B.sh ee_v1
# Standalone; run.sh-style GPU and PORT (defaults below).

GPUS=(${GPU:-7})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${PORT:-29575}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE                   --nnodes $NNODES                   --node_rank $NODE_RANK                   --master_addr $MASTER_ADDR                   --master_port $MASTER_PORT"

DATA=${1:-ee_v1}
BASE_PATH=.
# switches
KD_SCHEDULE=${KD_SCHEDULE:-anneal}
KD_RATIO=${KD_RATIO:-0.9}
KD_TYPE=${KD_TYPE:-fkl}
W_SPAN=${W_SPAN:-0}
TEACHERS=${TEACHERS:-}
if [ -z "${TEACHERS}" ]; then
    for d in ace geneva maven rams; do
        lora=$(python tools/best_lora.py ${BASE_PATH}/results/qwen3/general_domain/sft_4B_${DATA}_${d}) || exit 1
        TEACHERS+="${TEACHERS:+,}${d}=${lora}"
    done
fi
echo "teachers: ${TEACHERS}"
# model
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
[ -d models/Qwen3-0.6B ] && CKPT="models/Qwen3-0.6B"   # local copy from download.txt
TEACHER_CKPT_NAME="qwen3-4B"
TEACHER_CKPT="Qwen/Qwen3-4B-Instruct-2507"
[ -d models/Qwen3-4B-Instruct-2507 ] && TEACHER_CKPT="models/Qwen3-4B-Instruct-2507"
# data
DATA_DIR="${BASE_PATH}/processed_data/general_domain/${DATA}/qwen/"
# hp: as the SFT student (LR, 5 epochs, effective batch 32), but micro-batch 8 x GRAD_ACC 4 instead of
# 16 x 2: the teacher forward and the full-vocabulary KL need room on one 80GB GPU
BATCH_SIZE=8
LR=0.00005
GRAD_ACC=4
EVAL_BATCH_SIZE=128
EPOCHS=5
# length
MAX_LENGTH=768
# runtime
RUN=${KD_TYPE}_${KD_SCHEDULE}$([ "${KD_SCHEDULE}" = constant ] && echo "_kd${KD_RATIO}")_span${W_SPAN}
SAVE_PATH="${BASE_PATH}/results/qwen3/general_domain/bam_0.6B_4B_${DATA}/${RUN}"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-peft-paths ${TEACHERS}"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# score each source dataset in its own labels; dev answers every epoch, test answers when the mean
# per-dataset trigger F1 on dev improves
OPTS+=" --label-map ${BASE_PATH}/data/general_domain/ee_label_map.tsv"
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
OPTS+=" --kd-schedule ${KD_SCHEDULE}"
OPTS+=" --kd-ratio ${KD_RATIO}"
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
# type: KD loss on the logits (no student generation)
OPTS+=" --type ${KD_TYPE}"
# span loss
OPTS+=" --w-span-loss ${W_SPAN}"
OPTS+=" --teacher_layer_mapping 33 36"
OPTS+=" --student_layer_mapping 25 28"
# gen: greedy (no --do-sample)


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_finetune.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
