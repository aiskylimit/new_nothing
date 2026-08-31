#!/usr/bin/env bash
set -e


source /mnt/local/uvenvs/talas-vlm-embed/bin/activate

# python fix_lib.py

# #
# # 3. Unzip the dataset
# #
# mkdir -p vlm2vec_train/MMEB-train/images
# mkdir -p eval_images
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/N24News.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/images.zip -d ./eval_images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/DocVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/ChartQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/MSCOCO.zip -d ./vlm2vec_train/MMEB-train/images/

# #
# # 4. Unzip the cache
# #
# tar -xzf /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/B3_Qwen2_2B_cls.tar.gz -C .




# CUDA_VISIBLE_DEVICES=0 bash scripts/train_distill_talas_jepa_cls.sh &
# wait


# =========================
# 8. Eval
# =========================
# Run 4 eval scripts in parallel for each batch size, each one on a different GPU.

# CUDA_VISIBLE_DEVICES=0 bash eval_0.sh &
# wait

CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 0.5 &
CUDA_VISIBLE_DEVICES=1 bash script_full/train_distill_sigreg_cls.sh 1 1 1 0 1.0 0.05 &
CUDA_VISIBLE_DEVICES=2 bash script_full/train_distill_sigreg_cls.sh 1 1 0 1 1.0 0.05 &
CUDA_VISIBLE_DEVICES=3 bash script_full/train_distill_sigreg_cls.sh 1 0 1 1 1.0 0.05 &
CUDA_VISIBLE_DEVICES=4 bash script_full/train_distill_sigreg_cls.sh 0 1 1 1 1.0 0.05 &
CUDA_VISIBLE_DEVICES=5 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 0.01 &
CUDA_VISIBLE_DEVICES=6 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 2.0 0.05 &
CUDA_VISIBLE_DEVICES=7 bash script_full/train_distill_sigreg_cls.sh 1 1 0 0 1.0 0.05 &
wait

# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
