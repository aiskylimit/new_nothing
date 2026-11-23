#!/usr/bin/env bash
set -e
source /mnt/local/uvenvs/mrl-text/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")"

# Downloaded folders: bert-base-uncased, bge-m3, gte-Qwen2-1.5B-instruct.
# Set MRL_MODELS_DIR when models are stored outside this project.
MODEL_NAME="${MODEL_NAME:-${MRL_MODELS_DIR:-$PWD/models}/bge-m3}"
CUDA_DEVICE=2
EPOCHS=1
BATCH_SIZE=128
LR=2e-5
TEMPERATURE=0.05
SEED=42

declare -A TRAIN_DIRS=(
    [mrl]=MRL [mipic]=MIPIC [flag_mrl]=FlagMRL [ese]=ESE
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

for METHOD in mrl flag_mrl ese mipic; do
    OUTPUT_DIR="checkpoints/$METHOD/${MODEL_NAME##*/}/bs${BATCH_SIZE}_lr${LR}_t${TEMPERATURE}_ep${EPOCHS}_s${SEED}"
    EXTRA_ARGS=()
    if [[ "$METHOD" == flag_mrl ]]; then EXTRA_ARGS=(--flag_lr 0.05); fi
    echo "Running $METHOD -> $OUTPUT_DIR"

    "${PYTHON_BIN:-python}" -u "${TRAIN_DIRS[$METHOD]}/train.py" \
        --model_name "$MODEL_NAME" --output_dir "$OUTPUT_DIR" \
        --device cuda:0 \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" \
        --seed "$SEED" --temperature "$TEMPERATURE" \
        --eval_batch_size 512 \
        "${EXTRA_ARGS[@]}"
done
