#!/bin/bash

set -e

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
MASTER_PORT=${2-2012}
DEVICE=${CUDA_VISIBLE_DEVICES:-${DEVICE:-0}}
export CUDA_VISIBLE_DEVICES=${DEVICE}
STUDENT_PATH=${STUDENT_PATH:-"${BASE_PATH}/results/gpt2/train/init/gpt2-base"}
TEACHER_PATH=${TEACHER_PATH:-"${BASE_PATH}/results/gpt2/train/sft/gpt2-xlarge"}
VELOCITY_PATH="${BASE_PATH}/results/gpt2/train/velocity_field_t0"
ckpt="contra_t0/distillm/0.1B_1.5B"

# Download the pretrained teacher SFT and student initialization before running.
for checkpoint in "${STUDENT_PATH}" "${TEACHER_PATH}"
do
    if [ ! -f "${checkpoint}/config.json" ]; then
        echo "Missing downloaded checkpoint: ${checkpoint}/config.json" >&2
        exit 1
    fi
done

# Each experiment loads its own copy of the prepared field/projector.
for checkpoint in "${VELOCITY_PATH}/velocity_field.pth" "${VELOCITY_PATH}/projector.pth"
do
    if [ ! -f "${checkpoint}" ]; then
        echo "Missing pretrained field/projector: ${checkpoint}. Run runs/gpt2/run_t0.sh first." >&2
        exit 1
    fi
done

# Setup and shared offline-field pretraining run once in runs/gpt2/run_t0.sh.

# Train with +Contra at t=0
bash ${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} 1 \
    --model-path ${STUDENT_PATH} \
    --teacher-model-path ${TEACHER_PATH} \
    --velocity-field-path ${VELOCITY_PATH}/velocity_field.pth \
    --projector-path ${VELOCITY_PATH}/projector.pth \
    --save ${BASE_PATH}/results/gpt2/train/${ckpt}

# evaluation
EVAL_PORT=${EVAL_PORT:-2040}

for benchmark in dolly self_inst vicuna sinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh ${BASE_PATH} ${EVAL_PORT} 1 ${ckpt} --seed $seed --eval-batch-size 64
    done
done
