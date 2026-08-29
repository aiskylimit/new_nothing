#!/bin/bash
# SEGD train interface (cls). Override any flag via env vars, or extra CLI args:
#   EXP_NAME=foo LEARNING_RATE=2e-4 bash scripts/cls/train_distill_segd_cls.sh
#   bash scripts/cls/train_distill_segd_cls.sh --learning_rate 2e-4

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_distill_ddp.py}"

EXP_NAME="${EXP_NAME:-FastVLM-0.5B_segd_eos_cls}"
MODEL_NAME="${MODEL_NAME:-apple/FastVLM-0.5B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-raghavlite/B3_Qwen2_2B}"
LORA="${LORA:-True}"
TEACHER_LORA="${TEACHER_LORA:-True}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TEACHER_LORA_R="${TEACHER_LORA_R:-8}"
TEACHER_POOLING="${TEACHER_POOLING:-eos}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-qwen2_vl}"
MODEL_BACKBONE="${MODEL_BACKBONE:-llava_qwen2}"
POOLING="${POOLING:-eos}"
DATASET_NAME="${DATASET_NAME:-TIGER-Lab/MMEB-train}"
SUBSET_NAME="${SUBSET_NAME:-ImageNet_1K N24News HatefulMemes VOC2007 SUN397}"
DATASET_SPLIT="${DATASET_SPLIT:-original}"
IMAGE_DIR="${IMAGE_DIR:-vlm2vec_train/MMEB-train}"
PERCENT_DATA="${PERCENT_DATA:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-training/$EXP_NAME}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SEED="${SEED:-42}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NORMALIZE="${NORMALIZE:-True}"
TEACHER_NORMALIZE="${TEACHER_NORMALIZE:-True}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
KD_LOSS_TYPE="${KD_LOSS_TYPE:-segd_loss}"
SEGD_LAMBDA_SIM="${SEGD_LAMBDA_SIM:-1.0}"
SEGD_LAMBDA_SPECTRAL="${SEGD_LAMBDA_SPECTRAL:-1.0}"
SEGD_TAU_GRAPH="${SEGD_TAU_GRAPH:-1.0}"
SEGD_NUM_ALIGN_LAYERS="${SEGD_NUM_ALIGN_LAYERS:-4}"
SEGD_K_EIGEN="${SEGD_K_EIGEN:-0}"
SEGD_K_EIGEN_MIN="${SEGD_K_EIGEN_MIN:-8}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-low}"
NUM_PROJECTORS="${NUM_PROJECTORS:-1}"
PROJECTOR_LR="${PROJECTOR_LR:-5e-5}"
REPORT_TO="${REPORT_TO:-None}"

# shellcheck disable=SC2206
SUBSETS=($SUBSET_NAME)

torchrun --standalone \
    --nproc_per_node="$NUM_GPUS_PER_NODE" "$TRAIN_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --teacher_model_name "$TEACHER_MODEL_NAME" \
    --lora "$LORA" \
    --teacher_lora "$TEACHER_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --teacher_lora_r "$TEACHER_LORA_R" \
    --teacher_pooling "$TEACHER_POOLING" \
    --teacher_backbone "$TEACHER_BACKBONE" \
    --model_backbone "$MODEL_BACKBONE" \
    --pooling "$POOLING" \
    --dataset_name "$DATASET_NAME" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "$DATASET_SPLIT" \
    --image_dir "$IMAGE_DIR" \
    --percent_data "$PERCENT_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --bf16 \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy "$SAVE_STRATEGY" \
    --seed "$SEED" \
    --weight_decay "$WEIGHT_DECAY" \
    --normalize "$NORMALIZE" \
    --teacher_normalize "$TEACHER_NORMALIZE" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --kd_loss_type "$KD_LOSS_TYPE" \
    --segd_lambda_sim "$SEGD_LAMBDA_SIM" \
    --segd_lambda_spectral "$SEGD_LAMBDA_SPECTRAL" \
    --segd_tau_graph "$SEGD_TAU_GRAPH" \
    --segd_num_align_layers "$SEGD_NUM_ALIGN_LAYERS" \
    --segd_k_eigen "$SEGD_K_EIGEN" \
    --segd_k_eigen_min "$SEGD_K_EIGEN_MIN" \
    --image_resolution "$IMAGE_RESOLUTION" \
    --num_projectors "$NUM_PROJECTORS" \
    --projector_lr "$PROJECTOR_LR" \
    --report_to "$REPORT_TO" \
    "$@"
