#!/usr/bin/env bash
set -e

source /mnt/local/uvenvs/vlm-gvendi/bin/activate


# bash set_base_model_path.sh
# # python -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip/en_core_web_sm.zip').extractall('.')"
# python fix_lib.py

# # # #
# # # 3. Unzip the dataset
# # #
# mkdir -p vlm2vec_train/MMEB-train/images
# mkdir -p eval_images
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
# unzip ./vlm2vec_external/flickr30k/flickr30k-images.zip -d ./vlm2vec_external/flickr30k/

# # # #
# # # # 4. Unzip the cache
# # # #
# tar -xzf ./datasets/B3_Qwen2_2B_cls.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_2B_vqa.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_2B_grounding.tar.gz -C .
# tar -xzf ./datasets/B3_Qwen2_7B_cls.tar.gz -C .

# tar -xzf ./datasets/Qwen3VL_8B_cls.tar.gz -C .
# tar -xzf ./datasets/Qwen3VL_8B_vqa.tar.gz -C .
# tar -xzf ./datasets/Qwen3VL_8B_grounding.tar.gz -C .

CUDA_VISIBLE_DEVICES=0 bash ./scripts/pga/train_distill_pga_B3-Qwen2-7B_FastVLM-0.5B_grounding.sh &
CUDA_VISIBLE_DEVICES=1 bash ./scripts/pga/train_distill_pga_B3-Qwen2-7B_Llava-OV-0.5B_vqa.sh &
CUDA_VISIBLE_DEVICES=2 bash ./scripts/pga/train_distill_pga_B3-Qwen2-7B_Llava-OV-0.5B_grounding.sh & 
CUDA_VISIBLE_DEVICES=3 bash ./scripts/pga2/train_distill_pga2_Qwen3-VL-8B-Instruct_FastVLM-0.5B_cls.sh &
CUDA_VISIBLE_DEVICES=3 bash ./scripts/pga2/train_distill_pga2_Qwen3-VL-8B-Instruct_FastVLM-0.5B_vqa.sh &
CUDA_VISIBLE_DEVICES=3 bash ./scripts/pga2/train_distill_pga2_Qwen3-VL-8B-Instruct_FastVLM-0.5B_grounding.sh &

wait

# =========================
# 9. Copy JSON eval outputs
# =========================

JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"

python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
