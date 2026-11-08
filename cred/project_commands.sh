#!/usr/bin/env bash
set -e


source /mnt/local/uvenvs/cred/bin/activate


bash set_base_model_path.sh
# python -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip/en_core_web_sm.zip').extractall('.')"
python fix_lib.py

#
# 3. Unzip the dataset
#
mkdir -p eval_images
mkdir -p vlm2vec_train/MMEB-train/images
unzip ./vlm2vec_train/MMEB-train/images_zip/ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/N24News.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_eval/MMEB-eval/images.zip -d ./eval_images/
unzip ./vlm2vec_train/MMEB-train/images_zip/OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/DocVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/ChartQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/MSCOCO.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/VisDial.zip -d  ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/CIRR.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/VisualNews_t2i.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/VisualNews_i2t.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/MSCOCO_t2i.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/MSCOCO_i2t.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/NIGHTS.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./vlm2vec_train/MMEB-train/images_zip/WebQA.zip -d ./vlm2vec_train/MMEB-train/images/


# #
# # 4. Unzip the cache
# #
# tar -xzf ./vlm2vec_train/MMEB-train/B3_Qwen2_2B_cls.tar.gz -C .
# tar -xzf ./vlm2vec_train/MMEB-train/B3_Qwen2_2B_vqa.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_2B_grounding.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_7B_cls.tar.gz -C .


CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh  1 1 1 1 1.0 0.05 256 0.02 "" 5 none pr &
CUDA_VISIBLE_DEVICES=1 bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.05 256 0.02 "" 0 12 pr &
CUDA_VISIBLE_DEVICES=2 bash ./script_full/train_distill_sigreg_cls_pr.sh &
CUDA_VISIBLE_DEVICES=2 bash ./script_full/train_distill_sigreg_llava_ov_cls_bz16_2.sh 1 1 1 0 1.0 2.0 256 0.02 &


# bash run_time_mem.sh &

# CUDA_VISIBLE_DEVICES=4 bash ./script_full/train_distill_sigreg_cls_pr.sh &
# CUDA_VISIBLE_DEVICES=5 bash ./script_full/train_distill_sigreg_cls_abl.sh &
# CUDA_VISIBLE_DEVICES=6 bash ./script_full/train_distill_sigreg_llava_ov_cls_bz16_2.sh 1 1 1 1 1.0 2.0 256 0.02 &

wait

# CUDA_VISIBLE_DEVICES=7 bash script_full/infer_erank_analyze_cls.sh 1 1 1 0 1.0 1.0 256 0.07 &
# CUDA_VISIBLE_DEVICES=2 bash script_full/infer_erank_analyze_cls.sh 1 1 1 1 1.0 0.5 5 0.02 &
# CUDA_VISIBLE_DEVICES=7 bash script_full/infer_erank_analyze_cls_ov.sh 1 1 1 1 1.0 1.0 256 0.02 &
# CUDA_VISIBLE_DEVICES=6 bash script_full/infer_erank_analyze_cls_ov.sh 1 1 1 1 1.0 2.0 256 0.02 &
# CUDA_VISIBLE_DEVICES=7 bash ./script_full/infer_erank_analyze_cls_2.sh

# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json-v5}"

python json_filter.py ./MMEB-eval_outputs_v5 "${JSON_FILTER_DESTINATION}" --overwrite
