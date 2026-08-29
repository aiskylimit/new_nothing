#!/bin/bash

NUM_GPUS_PER_NODE=4
TRAIN_SCRIPT="train_centroid_ddp.py"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --teacher_lora True \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --teacher_normalize True \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --image_resolution "low" \
    --output_dir "training/B3_Qwen2_2B_centroid_cls" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --bf16 \
    --save_strategy "epoch" \
    --logging_steps 1 \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.03 \
    --num_centroids 64 \
    --centroid_hidden_dim 256 \
    --centroid_dim 128 \
    --centroid_layer_idx -1 \
    --centroid_temperature 0.02 \
    --sinkhorn_epsilon 0.05 \
    --sinkhorn_iters 5 \
    --drop_special_tokens False \
    --report_to None
