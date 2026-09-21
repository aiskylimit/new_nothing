#!/usr/bin/env bash
set -e

source .venv/bin/activate


bash set_base_model_path.sh
python -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip/en_core_web_sm.zip').extractall('.')"
python fix_lib.py

#
# 3. Unzip the dataset
#
mkdir -p vlm2vec_train/MMEB-train/images
mkdir -p eval_images
unzip ./datasets/ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/N24News.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/images.zip -d ./eval_images/
unzip ./datasets/OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/DocVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/ChartQA.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/MSCOCO.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/RefCOCO.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/RefCOCO-Matching.zip -d ./vlm2vec_train/MMEB-train/images/
unzip ./datasets/Visual7W-Pointing.zip -d ./vlm2vec_train/MMEB-train/images/

#
# 4. Unzip the cache
#
tar -xzf ./datasets/B3_Qwen2_2B_cls.tar.gz -C .
tar -xzf ./datasets/B3_Qwen2_2B_vqa.tar.gz -C .
tar -xzf ./datasets/B3_Qwen2_2B_grounding.tar.gz -C .
tar -xzf ./datasets/B3_Qwen2_7B_cls.tar.gz -C .