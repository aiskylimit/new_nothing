#d
#datasets
--url https://huggingface.co/datasets/datht/processed-cl-ace/resolve/main/ace_all.tar.gz /mnt/local/@PROJECT@/OpenED/ace_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-maven/resolve/main/maven_all.tar.gz /mnt/local/@PROJECT@/OpenED/maven_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-rams/resolve/main/rams_all.tar.gz /mnt/local/@PROJECT@/OpenED/rams_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-geneva/resolve/main/geneva_all.tar.gz /mnt/local/@PROJECT@/OpenED/geneva_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-tacred/resolve/main/tacred_all.tar.gz /mnt/local/@PROJECT@/OpenED/tacred_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-fewrel/resolve/main/fewrel_all.tar.gz /mnt/local/@PROJECT@/OpenED/fewrel_all.tar.gz
#models
--hf Qwen/Qwen3-0.6B /mnt/local/@PROJECT@/OpenED/models/Qwen3-0.6B

#opened
#v1

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/logs/

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
kill -9 497 498 499 500
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
bash ./project_commands.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./offline_olmo7b_b200
# bash ./project_commands.sh
