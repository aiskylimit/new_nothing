#!/usr/bin/env bash
set -e


source /mnt/local/uvenvs/talas-vlm-embed/bin/activate


# bash set_base_model_path.sh
# # python -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip/en_core_web_sm.zip').extractall('.')"
# python fix_lib.py

# # #
# # # 3. Unzip the dataset
# # #
# mkdir -p eval_images
# mkdir -p vlm2vec_train/MMEB-train/images
# unzip ./vlm2vec_train/MMEB-train/images_zip/ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/N24News.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_eval/MMEB-eval/images.zip -d ./eval_images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/DocVQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/ChartQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/MSCOCO.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/VisDial.zip -d  ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/CIRR.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/VisualNews_t2i.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/VisualNews_i2t.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/MSCOCO_t2i.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/MSCOCO_i2t.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/NIGHTS.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip ./vlm2vec_train/MMEB-train/images_zip/WebQA.zip -d ./vlm2vec_train/MMEB-train/images/


# # #
# # # 4. Unzip the cache
# # #
# tar -xzf ./datasets/B3_Qwen2_2B_cls.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_2B_vqa.tar.gz -C .
# # tar -xzf ./datasets/B3_Qwen2_2B_grounding.tar.gz -C .
# # tar -xzf ./datasets/B3_Qwen2_7B_cls.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_2B_ret.tar.gz -C .
# # tar -xzf ./datasets/Qwen3VL_8B_cls.tar.gz -C .
# tar -xzf gmm.tar.gz -C .


# CUDA_VISIBLE_DEVICES=4 bash ./baseline_scripts/train_distill_pga_cls.sh &
# CUDA_VISIBLE_DEVICES=5 bash ./baseline_scripts/train_distill_pga_vqa.sh &
# CUDA_VISIBLE_DEVICES=6 bash ./baseline_scripts/train_distill_ov_pga_cls.sh &
# CUDA_VISIBLE_DEVICES=7 bash ./baseline_scripts/train_distill_ov_pga_vqa.sh &
# CUDA_VISIBLE_DEVICES=2 bash ./baseline_scripts/train_distill_pga_qwen2-7B_cls.sh &
# CUDA_VISIBLE_DEVICES=3 bash ./baseline_scripts/train_distill_pga_grounding.sh &
# CUDA_VISIBLE_DEVICES=7 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh
CUDA_VISIBLE_DEVICES=0 bash ./scripts/train_distill_talas_jepa_ret_CIRR.sh 1 1 10 0.1 17 5 24 29536 &
CUDA_VISIBLE_DEVICES=1 bash ./scripts/train_distill_talas_jepa_vqa.sh 1 0 10 0.1 17 5 24 29515 &
CUDA_VISIBLE_DEVICES=2 bash ./scripts/train_distill_talas_jepa_vqa.sh 0 1 10 0.1 17 5 24 29514 &
CUDA_VISIBLE_DEVICES=3 bash ./scripts/train_distill_talas_jepa_vqa.sh 1 1 10 0.3 17 5 24 29514 &
CUDA_VISIBLE_DEVICES=4 bash ./scripts/train_distill_talas_jepa_vqa.sh 1 1 10 0.5 17 5 24 29514 &
CUDA_VISIBLE_DEVICES=5 bash ./scripts/train_distill_talas_jepa_vqa.sh 1 1 10 1.0 17 5 24 29514 &

# ov cls đang chạy với sw=0.2, numt=29
CUDA_VISIBLE_DEVICES=0 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 0 10 0.1 29 5 24 29515 &
CUDA_VISIBLE_DEVICES=1 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 1 10 0.1 29 5 24 29515 &
CUDA_VISIBLE_DEVICES=2 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 1 10 0.2 29 5 24 29514 &
CUDA_VISIBLE_DEVICES=3 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 1 10 0.3 29 5 24 29514 &
CUDA_VISIBLE_DEVICES=4 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 1 10 0.5 29 5 24 29514 &
CUDA_VISIBLE_DEVICES=5 bash ./scripts/train_distill_talas_jepa_llava-ov_cls.sh 1 1 10 1.0 29 5 24 29514 &

# Còn InfoNCE + relational + một distribution-matching baseline đơn giản MMD, CORAL, or direct Sliced Wasserstein after PCA/random projection)

wait

# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
