#d
#datasets
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/llava_v1_5_mix665k.json /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/train_val_images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/textvqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/coco/train2017.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/coco
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/gqa/images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/gqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/ocr_vqa/ocr_vqa_images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/ocr_vqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/ocr_vqa/dataset.json /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/ocr_vqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/vg/images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/vg
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/vg/images2.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/vg
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/GQA_TestDev_Balanced.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MME.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/RealWorldQA.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/ScienceQA_TEST.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/AI2D_TEST_NO_MASK.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MMMU_DEV_VAL.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MMStar.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/ChartQA_TEST.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/DocVQA_VAL.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/TextVQA_VAL.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/OCRBench.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
#models
--hf KamilaMila/FastVLM-0.5B /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/KamilaMila/FastVLM-0.5B
--hf Qwen/Qwen2-VL-7B-Instruct /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models/Qwen/Qwen2-VL-7B-Instruct

#vlm
#v1

# cd SpectralGuidedLearning && GPUS=5 bash project_commands_b200_gain.sh 

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/pl_logs/
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/collected_logs.tar.gz +a

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 2
nvidia-smi

# ls OPSD/results/raw/olmo3-7b-think/grpo/

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./multi-mode-distill
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh

# cd ./cypher-extract
# bash ./project_command.sh

# cd ./OpenED
# bash gather_pl_logs.sh 
# bash gather_logs.sh all35
# tar -czf collected_logs.tar.gz collected_logs/all35/
# ls -lh collected_logs.tar.gz
# bash ./project_commands_19.sh

# cd ./opsd
# bash ./project_commands.sh

