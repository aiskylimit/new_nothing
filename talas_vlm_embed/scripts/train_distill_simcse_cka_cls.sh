#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_distill_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name models/FastVLM-0.5B \
    --teacher_model_name "models/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "vlm2vec_train/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/FastVLM-0.5B_simcse_cka_cls" \
    --per_device_train_batch_size 16 \
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
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --kd_weight 1.0 \
    --kd_loss_type "simcse_cka_kd_loss" \
    --image_resolution "low" \
    --num_self_kd_layers 3 \
    --projector_lr 5e-5 \
    --report_to None


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
  --model_name "training/FastVLM-0.5B_simcse_cka_cls/checkpoint-epoch-0" \
  --encode_output_path "./MMEB-eval_outputs_v3/FastVLM-0.5B_simcse_cka_cls" \
  --lora True \
  --lora_r 64 \
  --lora_alpha 64 \
  --pooling eos \
  --model_backbone llava_qwen2 \
  --normalize True \
  --bf16 \
  --dataset_name vlm2vec_eval/MMEB-eval \
  --subset_name "${EVAL_SUBSETS[@]}" \
  --dataset_split test \
  --per_device_eval_batch_size 16 \
  --image_dir eval_images/ \
  --tgt_prefix_mod \
  --load_pretrained_lora True \
  --report_to none