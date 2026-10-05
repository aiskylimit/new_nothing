#d
#datasets
--hf-dataset simplescaling/s1K-1.1 /mnt/local/_data/@PROJECT@/s1K-1.1
--hf-dataset Maxwell-Jia/AIME_2024 /mnt/local/_data/@PROJECT@/AIME_2024
--hf-dataset yentinglin/aime_2025 /mnt/local/_data/@PROJECT@/aime_2025
--hf-dataset HuggingFaceH4/MATH-500 /mnt/local/_data/@PROJECT@/MATH-500
--hf-dataset AI-MO/aimo-validation-amc /mnt/local/_data/@PROJECT@/aimo-validation-amc
#models
--hf Qwen/Qwen2.5-7B-Instruct /mnt/local/_models/@PROJECT@/Qwen2.5-7B-Instruct
--hf Qwen/Qwen3-8B /mnt/local/_models/@PROJECT@/Qwen3-8B

#log
#v1

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/logs/
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/collected_logs/thiennh-p6-tw7m-worker-0_20261005_1624/

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 2
nvidia-smi

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

# cd ./cypher-extract
# bash ./project_command.sh

cd ./OpenED
bash gather_logs.sh
# bash ./project_commands_3.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./offline_olmo7b_b200
# bash ./project_commands.sh
