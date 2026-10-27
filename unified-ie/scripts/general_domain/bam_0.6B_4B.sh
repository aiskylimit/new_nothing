#! /bin/bash
# Multi-teacher KD as in BAM (Clark et al. 2019): the Qwen3-0.6B student (full fine-tuning) learns from one
# Qwen3-4B LoRA teacher per source dataset, each row distilled from the teacher of its own dataset.
#   bash scripts/general_domain/bam_0.6B_4B.sh mt_v2   # EE + NER + RE in one student, 8 teachers
#   bash scripts/general_domain/bam_0.6B_4B.sh ee_v1   # EE only (ee_v1 or ee_v2), 4 teachers
# Same data and hyperparameters as the SFT student of that folder (sft_student_0.6B_mt_v2.sh,
# sft_student_0.6B_ee_v*.sh), so that student is the baseline.
# Each part is a switch (env variable, default = BAM); the run dir is named after them:
#   KD_SCHEDULE=anneal    BAM teacher annealing (KD weight 1 -> 0); constant = fixed KD_RATIO
#   KD_TYPE=fkl           forward KL, BAM's loss; sfkl = skewed forward KL of EventKD
#   W_SPAN=0              span loss off; 1.0 = EventKD span loss on teacher layers 33/36 (EE rows only:
#                         NER and RE rows have no spans yet)
#   SUBSET=               all train rows; 0.5 = fixed square-root subset (same rows as the SFT student's SUBSET=0.5)
#   TEACHERS=             name=lora_dir,...; empty = best dev epoch of each teacher run in results/ below
# e.g. KD_SCHEDULE=constant KD_TYPE=sfkl W_SPAN=1.0 bash scripts/general_domain/bam_0.6B_4B.sh ee_v1
# Standalone; run.sh-style GPU and PORT (defaults below).

GPUS=(${GPU:-7})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${PORT:-29575}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE                   --nnodes $NNODES                   --node_rank $NODE_RANK                   --master_addr $MASTER_ADDR                   --master_port $MASTER_PORT"

DATA=${1:-mt_v2}
BASE_PATH=.
MAPS=${BASE_PATH}/data/general_domain
# per data folder: teacher runs (dataset=results dir), label maps, prompt width
case ${DATA} in
    ee_v1|ee_v2)
        TEACHER_RUNS="ace=sft_4B_${DATA}_ace geneva=sft_4B_${DATA}_geneva maven=sft_4B_${DATA}_maven rams=sft_4B_${DATA}_rams"
        LABEL_MAP=${MAPS}/ee_label_map.tsv
        MAX_PROMPT=460;;
    mt_v2)
        TEACHER_RUNS="ace=sft_4B_ee_v2_ace geneva=sft_4B_ee_v2_geneva maven=sft_4B_ee_v2_maven rams=sft_4B_ee_v2_rams"
        TEACHER_RUNS+=" conll03=sft_4B_ner_v2_conll03 ontonotes=sft_4B_ner_v2_ontonotes"
        TEACHER_RUNS+=" tacred=sft_4B_re_tacred fewrel=sft_4B_re_fewrel"
        LABEL_MAP=${MAPS}/ee_label_map.tsv,${MAPS}/ner_label_map.tsv
        MAX_PROMPT=480;;
    *)
        echo "unknown data folder ${DATA} (ee_v1, ee_v2, mt_v2)"
        exit 1;;
esac
# switches
KD_SCHEDULE=${KD_SCHEDULE:-anneal}
KD_RATIO=${KD_RATIO:-0.9}
KD_TYPE=${KD_TYPE:-fkl}
W_SPAN=${W_SPAN:-0}
SUBSET=${SUBSET:-}
TEACHERS=${TEACHERS:-}
if [ -z "${TEACHERS}" ]; then
    for t in ${TEACHER_RUNS}; do
        lora=$(python tools/best_lora.py ${BASE_PATH}/results/qwen3/general_domain/${t#*=}) || exit 1
        TEACHERS+="${TEACHERS:+,}${t%%=*}=${lora}"
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
# data (folders added after the first processed_data.zip, like mt_v2, come out of the zip here)
DATA_DIR="${BASE_PATH}/processed_data/general_domain/${DATA}/qwen/"
[ -f ${DATA_DIR}/train_0.bin ] || python tools/unpack_data.py ${DATA} || exit 1
# hp: as the SFT student (LR, epochs, effective batch 32), but micro-batch 8 x GRAD_ACC 4 instead of
# 16 x 2: the teacher forward and the full-vocabulary KL need room on one 80GB GPU
BATCH_SIZE=8
LR=0.00005
GRAD_ACC=4
EVAL_BATCH_SIZE=128
EPOCHS=5
# length
MAX_LENGTH=768
# runtime
RUN=${KD_TYPE}_${KD_SCHEDULE}$([ "${KD_SCHEDULE}" = constant ] && echo "_kd${KD_RATIO}")_span${W_SPAN}${SUBSET:+_subset${SUBSET}}
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
[ -n "${SUBSET}" ] && OPTS+=" --subset-alpha ${SUBSET}"
# score each source dataset in its own labels; dev answers every epoch, test answers when the dev score
# (mean per-dataset F1; for mt_v2 the mean of the EE, NER and RE scores) improves
OPTS+=" --label-map ${LABEL_MAP}"
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
OPTS+=" --max-prompt-length ${MAX_PROMPT}"
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
