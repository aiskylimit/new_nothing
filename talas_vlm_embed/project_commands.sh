#!/usr/bin/env bash
set -e


source /mnt/local/uvenvs/talas-vlm-embed/bin/activate


# bash set_base_model_path.sh
# python -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip/en_core_web_sm.zip').extractall('.')"
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
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/DocVQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/ChartQA.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
# unzip /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/MSCOCO.zip -d ./vlm2vec_train/MMEB-train/images/

# #
# # 4. Unzip the cache
# #
# tar -xzf /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/B3_Qwen2_2B_cls.tar.gz -C .
# tar -xzf /mnt/local/aiskylimit_new_nothing/talas_vlm_embed/datasets/B3_Qwen2_2B_vqa.tar.gz -C .




# CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 1.0 9 0.02 &
# CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 0.5 9 0.02 &
# CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 0.3 9 0.02 &
# CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 0.3 9 0.05 &
# CUDA_VISIBLE_DEVICES=1 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 0.5 9 0.05 &
# CUDA_VISIBLE_DEVICES=1 bash script_full/train_distill_sigreg_cls.sh 1 1 1 1 1.0 1.0 9 0.05 &
# # # wait

# CUDA_VISIBLE_DEVICES=1 bash script_full/train_distill_sigreg_vqa.sh 1 1 1 1 1.0 1.0 9 0.02 &
# # # CUDA_VISIBLE_DEVICES=1 bash script_full/train_distill_sigreg_vqa.sh 1 1 1 1 1.0 0.2 &
# # # CUDA_VISIBLE_DEVICES=4 bash script_full/train_distill_sigreg_vqa.sh 1 1 1 1 1.0 0.01 &
# # # CUDA_VISIBLE_DEVICES=4 bash script_full/train_distill_sigreg_vqa.sh 1 1 1 1 1.0 0.1 &
# # # CUDA_VISIBLE_DEVICES=5 bash script_full/train_distill_sigreg_vqa.sh 1 1 1 1 1.0 0.2 &
# # # wait

# CUDA_VISIBLE_DEVICES=1 bash script_full/train_distill_sigreg_llava_ov_cls.sh 1 1 1 1 1.0 1.0 9 0.1 &
# # CUDA_VISIBLE_DEVICES=2 bash scripts/train_single_ov_eos_cls.sh &
# # CUDA_VISIBLE_DEVICES=3 bash script_full/train_distill_sigreg_llava_ov_vqa.sh 1 1 1 1 1.0 0.1 6 1&
# # CUDA_VISIBLE_DEVICES=6 bash script_full/train_distill_sigreg_llava_ov_cls.sh 1 1 1 1 1.0 0.05 &
# # CUDA_VISIBLE_DEVICES=7 bash script_full/train_distill_sigreg_llava_ov_cls.sh 1 1 1 1 1.0 0.01 &
# # CUDA_VISIBLE_DEVICES=7 bash script_full/train_distill_sigreg_llava_ov_cls.sh 1 1 1 1 1.0 0.2 &
# # CUDA_VISIBLE_DEVICES=5 bash script_full/train_distill_sigreg_llava_ov_cls.sh 1 1 1 1 1.0 0.5 &
# CUDA_VISIBLE_DEVICES=2 bash scripts/train_distill_span_propose_llava_ov_cls.sh &
# CUDA_VISIBLE_DEVICES=3 bash scripts/train_distill_span_propose_llava_ov_cls_2.sh &
# CUDA_VISIBLE_DEVICES=2 bash scripts/train_distill_simcse_cka_cls.sh &
# CUDA_VISIBLE_DEVICES=3 bash scripts/train_distill_simcse_cka_vqa.sh &

wait

CUDA_VISIBLE_DEVICES=2 bash script_full/infer_erank_analyze_cls.sh 1 1 1 1 1.0 1.0 9 0.02

# =========================
# 9. Copy JSON eval outputs
# =========================

# JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json-v3}"

# python json_filter.py ./MMEB-eval_outputs_v3 "${JSON_FILTER_DESTINATION}" --overwrite
