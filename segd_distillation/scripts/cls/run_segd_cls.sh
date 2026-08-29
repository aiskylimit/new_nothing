#!/bin/bash
# User-facing SEGD cls train wrapper.
#
# Usage:
#   bash scripts/cls/run_segd_cls.sh
#   LEARNING_RATE=2e-4 SEGD_LAMBDA_SPECTRAL=0.5 bash scripts/cls/run_segd_cls.sh
#   bash scripts/cls/run_segd_cls.sh --logging_steps 10
#
# Extra CLI args are forwarded to `train_distill_ddp.py`.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Hyperparams (defaults match scripts/cls/train_distill_segd_cls.sh)
# ---------------------------------------------------------------------------
export EXP_NAME="${EXP_NAME:-FastVLM-0.5B_segd_eos_cls}"

export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export PROJECTOR_LR="${PROJECTOR_LR:-5e-5}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export SEED="${SEED:-42}"

export SEGD_LAMBDA_SIM="${SEGD_LAMBDA_SIM:-1.0}"
export SEGD_LAMBDA_SPECTRAL="${SEGD_LAMBDA_SPECTRAL:-1.0}"
export SEGD_TAU_GRAPH="${SEGD_TAU_GRAPH:-1.0}"
export SEGD_NUM_ALIGN_LAYERS="${SEGD_NUM_ALIGN_LAYERS:-4}"
export SEGD_K_EIGEN="${SEGD_K_EIGEN:-0}"
export SEGD_K_EIGEN_MIN="${SEGD_K_EIGEN_MIN:-8}"

export LORA_R="${LORA_R:-64}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export NUM_PROJECTORS="${NUM_PROJECTORS:-1}"
export POOLING="${POOLING:-eos}"
export TEACHER_POOLING="${TEACHER_POOLING:-eos}"
export IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-low}"
export PERCENT_DATA="${PERCENT_DATA:-1.0}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"

bash scripts/cls/train_distill_segd_cls.sh "$@"

