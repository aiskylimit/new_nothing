#3
#talas-llava
#v2

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json/ +a

# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=6,7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 3
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


# cd ./talas_vlm_embed
# # CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &
# CUDA_VISIBLE_DEVICES=6,7 python3 multi_gpu.py &
# bash ./project_commands.sh
# # CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=6,7 python3 /tmp/llm_pretrain_burn.py &

# cd ./spectral-guided-learning
# bash ./project_commands.sh

# cd ./reward-guidance-main
# bash ./project_command.sh

# cd ./VLM_Distillation-main
# bash ./project_commands.sh
# CUDA_VISIBLE_DEVICES=4,5 python3 /tmp/llm_pretrain_burn.py &

