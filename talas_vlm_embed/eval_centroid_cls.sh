#!/bin/bash

NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}

EVAL_SCRIPT="eval_centroid_ddp.py"

CENTROID_CHECKPOINT="training/B3_Qwen2_2B_centroid_cls/checkpoint-final/centroid.pt"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $EVAL_SCRIPT \
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
    --output_dir "training/B3_Qwen2_2B_centroid_cls_eval" \
    --encode_output_path "./MMEB-eval_outputs/B3_Qwen2_2B_centroid_cls_eval/" \
    --per_device_eval_batch_size 8 \
    --num_centroids 64 \
    --centroid_hidden_dim 256 \
    --centroid_dim 128 \
    --centroid_layer_idx -1 \
    --centroid_temperature 0.02 \
    --sinkhorn_epsilon 0.05 \
    --sinkhorn_iters 5 \
    --drop_special_tokens False \
    --centroid_checkpoint "$CENTROID_CHECKPOINT" \
    --report_to None
