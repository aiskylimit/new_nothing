#!/bin/bash
# Eval SEGD student on MMEB-eval (classification).
# Usage:
#   bash scripts/cls/eval_segd_cls.sh
#   EXP_NAME=my_run bash scripts/cls/eval_segd_cls.sh
#   MODEL=training/.../checkpoint-final bash scripts/cls/eval_segd_cls.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_NAME="${EXP_NAME:-FastVLM-0.5B_segd_eos_cls}"
MODEL="${MODEL:-training/${EXP_NAME}/checkpoint-epoch-0}"
# Or: MODEL="training/${EXP_NAME}/checkpoint-final"

SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
  # OOD:
  # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
)

python eval_mmeb.py \
    --model_name "$MODEL" \
    --encode_output_path "./MMEB-eval_outputs/${EXP_NAME}/" \
    --lora True --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 64 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none

