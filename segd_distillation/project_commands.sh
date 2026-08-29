#!/usr/bin/env bash
set -e

# Run this file after:
#   cd talas_vlm_embed

# =========================
# 1. Optional system setup
# =========================
# Uncomment these lines if this is a fresh machine and you have sudo access.
#
# sudo apt-get update
# sudo apt-get upgrade -y


# =========================
# 2. Create Python env and install requirements
# =========================
# export UV_PROJECT_ENVIRONMENT=vlm
# uv sync
source vlm/bin/activate


# =========================
# 3. Optional eval images
# =========================
# README says this step is optional.
# Uncomment if you need eval images.

# wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
# unzip -o images.zip -d eval_images/
# rm -rf images.zip

# =========================
# 4. Optional train images
# =========================
# This can take more than 1 hour.
# Uncomment if you need train images.
#
# bash download_traindata.sh
# bash download_traindata_2.sh

# python download.py

# =========================
# 5. Optional teacher output
# =========================
# rm -rf caching
# hf download VoCuc/vlm-teacher-embedding \
#   B3_Qwen2_2B_cls.zip \
#   --repo-type dataset \
#   --local-dir .

# unzip -o B3_Qwen2_2B_cls.zip 

# hf download VoCuc/vlm-teacher-embedding \
#   B3_Qwen2_2B_vqa.zip \
#   --repo-type dataset \
#   --local-dir .

# unzip -o B3_Qwen2_2B_vqa.zip 


# =========================
# 6. Fix transformers code
# =========================
# README says this fixes the qwen2_vl image processor issue.
# python fix_lib.py


# =========================
# 7. Train
# =========================
# Before running, check these args in scripts/test_gvendi.sh:
#   --image_dir
#   --teacher_cache_dir
#
# Uncomment to start training.

CUDA_VISIBLE_DEVICES=0 bash scripts/train_distill_talas_cls.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/train_distill_talas_cls_1.sh &
CUDA_VISIBLE_DEVICES=2 bash scripts/train_distill_talas_cls_2.sh &
CUDA_VISIBLE_DEVICES=3 bash scripts/train_distill_talas_cls_3.sh &
wait



# =========================
# 8. Eval
# =========================
# Run 4 eval scripts in parallel for each batch size, each one on a different GPU.

CUDA_VISIBLE_DEVICES=0 bash eval_0.sh &
CUDA_VISIBLE_DEVICES=1 bash eval_1.sh &
CUDA_VISIBLE_DEVICES=2 bash eval_2.sh &
CUDA_VISIBLE_DEVICES=3 bash eval_3.sh &
wait

# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
