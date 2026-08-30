#1 +60
#talas_vlm_embed
#v1


# nvidia-smi
# CUDA_VISIBLE_DEVICES=0 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=1 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=2 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=3 python3 /tmp/llm_pretrain_burn.py &
CUDA_VISIBLE_DEVICES=4 python3 /tmp/llm_pretrain_burn.py &
CUDA_VISIBLE_DEVICES=5 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=6 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader)
# sleep 3
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd ./talas_vlm_embed
bash ./project_commands.sh

# cd ./spectral-guided-learning
# bash ./project_commands.sh

# cd ./reward-guidance-main
# bash ./project_command.sh

# cd ./VLM_Distillation-main
# bash ./project_commands.sh