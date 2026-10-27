#! /bin/bash
# SFT of the Qwen3-0.6B student (full fine-tuning, CE only) on all three tasks at once: mt_v2 = ee_v2 (ACE,
# MAVEN, RAMS, GENEVA) + ner_v2 (CoNLL-2003, OntoNotes) + TACRED + FewRel, 148.6k train rows, v2 labels.
# Baseline of the multi-teacher students. Dev/test are scored per task; the checkpoint is picked on the mean
# of the EE, NER and RE dev scores.
#   bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh
#   SUBSET=0.5 bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh    # square-root subset
# SUBSET=a: train on a fixed subset where dataset d keeps rows prop. to n_d^a, drawn once with the seed (the
# same rows in every run, BAM included); at 0.5 that's 41,183 of the 148,583 rows. Same 5 epochs over it, so
# no row is seen more often than in full-data training.
# Standalone: run.sh-style GPU and PORT (defaults below).

GPUS=(${GPU:-7})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${PORT:-29575}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE                   --nnodes $NNODES                   --node_rank $NODE_RANK                   --master_addr $MASTER_ADDR                   --master_port $MASTER_PORT"

DATA=mt_v2
BASE_PATH=.
SUBSET=${SUBSET:-}
# model
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
[ -d models/Qwen3-0.6B ] && CKPT="models/Qwen3-0.6B"   # local copy from download.txt
# data (mt_v2 came after the first processed_data.zip: an older unpack gets it from the zip here)
DATA_DIR="${BASE_PATH}/processed_data/general_domain/${DATA}/qwen/"
[ -f ${DATA_DIR}/train_0.bin ] || python tools/unpack_data.py ${DATA} || exit 1
# hp: one 80GB GPU (B200): micro-batch 16 x GRAD_ACC 2 = effective batch 32, as the per-task students
BATCH_SIZE=16
LR=0.00005
GRAD_ACC=2
EVAL_BATCH_SIZE=128
EPOCHS=5
# length
MAX_LENGTH=768
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3/general_domain/sft_0.6B_${DATA}${SUBSET:+_subset${SUBSET}}"
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
[ -n "${SUBSET}" ] && OPTS+=" --subset-alpha ${SUBSET}"
# score each source dataset in its own labels (EE and NER maps); dev answers every epoch, test answers when
# the mean of the three task scores on dev improves
OPTS+=" --label-map ${BASE_PATH}/data/general_domain/ee_label_map.tsv,${BASE_PATH}/data/general_domain/ner_label_map.tsv"
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
# length: NER prompts need 480
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
