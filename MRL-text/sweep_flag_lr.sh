#!/usr/bin/env bash
set -euo pipefail
source /mnt/local/uvenvs/mrl-text/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")"

CUDA_DEVICE="${CUDA_DEVICE:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Examples:
#   bash sweep_flag_lr.sh
#   BATCH_SIZE=64 LR=1e-5 EPOCHS=3 bash sweep_flag_lr.sh
#   CUDA_DEVICE=3 PRECISION=fp32 bash sweep_flag_lr.sh

# Training parameters
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-auto}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-42}"
DATA_PATH="${DATA_PATH:-$PWD/data/quora-duplicates-pair.csv}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-150000}"
MODEL_NAME="${MODEL_NAME:-${MRL_MODELS_DIR:-$PWD/models}/bert-base-uncased}"

# FlagMRL parameters
FLAG_OBJECTIVE="${FLAG_OBJECTIVE:-trace_ratio}"
FLAG_INIT="${FLAG_INIT:-identity}"
FLAG_EPS="${FLAG_EPS:-1e-4}"
FLAG_LAMBDA_NEG="${FLAG_LAMBDA_NEG:-1.0}"
FLAG_RETRACTION="${FLAG_RETRACTION:-svd}"
FLAG_UPDATE_EVERY="${FLAG_UPDATE_EVERY:-1}"
FLAG_LAMBDA_CROSS="${FLAG_LAMBDA_CROSS:-0.0}"
FLAG_FAR_MODE="${FLAG_FAR_MODE:-uniform}"
FLAG_FAR_TAU="${FLAG_FAR_TAU:-0.1}"
FLAG_INIT_BATCHES="${FLAG_INIT_BATCHES:-8}"

# Evaluation parameters
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-256}"

TRAIN_ARGS=(
    --model_name "$MODEL_NAME"
    --flag_objective "$FLAG_OBJECTIVE"
    --data_path "$DATA_PATH"
    --max_train_samples "$MAX_TRAIN_SAMPLES"
    --device "$DEVICE"
    --precision "$PRECISION"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_steps "$WARMUP_STEPS"
    --max_length "$MAX_LENGTH"
    --max_grad_norm "$MAX_GRAD_NORM"
    --temperature "$TEMPERATURE"
    --num_workers "$NUM_WORKERS"
    --log_every "$LOG_EVERY"
    --seed "$SEED"
    --flag_init "$FLAG_INIT"
    --flag_init_batches "$FLAG_INIT_BATCHES"
    --flag_eps "$FLAG_EPS"
    --flag_lambda_neg "$FLAG_LAMBDA_NEG"
    --flag_retraction "$FLAG_RETRACTION"
    --flag_update_every "$FLAG_UPDATE_EVERY"
    --flag_lambda_cross "$FLAG_LAMBDA_CROSS"
    --flag_far_mode "$FLAG_FAR_MODE"
    --flag_far_tau "$FLAG_FAR_TAU"
    --run_eval 1
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --eval_max_length "$EVAL_MAX_LENGTH"
    --eval_splits test
    --eval_languages eng
)

# Sweep only --flag_lr; model is fixed to BERT and method to FlagMRL.
for FLAG_LR in 0.01 0.03 0.05 0.1; do
    FLR_TAG="${FLAG_LR//./p}"
    OUTPUT_DIR="checkpoints/flag_mrl/${MODEL_NAME##*/}/bs${BATCH_SIZE}_lr${LR}_t${TEMPERATURE}_ep${EPOCHS}_s${SEED}_flr${FLR_TAG}"

    echo "================================================"
    echo "FlagMRL | model=bert | flag_lr=$FLAG_LR"
    echo "Output: $OUTPUT_DIR"
    echo "================================================"

    "$PYTHON_BIN" -u FlagMRL/train.py \
        "${TRAIN_ARGS[@]}" \
        --output_dir "$OUTPUT_DIR" \
        --flag_lr "$FLAG_LR"
done
