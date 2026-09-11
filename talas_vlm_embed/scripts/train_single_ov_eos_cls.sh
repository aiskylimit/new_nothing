#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_ddp.py"

# SUBSETS=(
#   "VOC2007"
#   "OK-VQA"
# )

SUBSETS=(
  "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
#   "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
)

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name "models/llava-onevision-qwen2-0.5b-ov-hf" \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_onevision" \
    --pooling "eos" \
    --dataset_name "vlm2vec_train/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "training/llava_ov-0.5B_eos_cls" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --kd_loss_type "contrastive" \
    --image_resolution "tiny" \
    --report_to "none" 



EVAL_SUBSETS=(
    "ImageNet-1K"
    "N24News"
    "HatefulMemes"
    "VOC2007"
    "SUN397"
    "Place365"
    "ImageNet-A"
    "ImageNet-R"
    "ObjectNet"
    "Country211"
)


python eval_mmeb.py \
  --model_name "training/llava_ov-0.5B_eos_cls/checkpoint-epoch-0" \
  --encode_output_path "./MMEB-eval_outputs_v3/llava_ov-0.5B_eos_cls" \
  --lora True \
  --lora_r 64 \
  --lora_alpha 64 \
  --pooling eos \
  --model_backbone llava_onevision \
  --normalize True \
  --bf16 \
  --dataset_name vlm2vec_eval/MMEB-eval \
  --subset_name "${EVAL_SUBSETS[@]}" \
  --dataset_split test \
  --per_device_eval_batch_size 2 \
  --image_dir eval_images/ \
  --image_resolution "low" \
  --tgt_prefix_mod \
  --load_pretrained_lora True \
  --report_to none
