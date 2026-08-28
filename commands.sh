#1 +30
#burn
#v1


# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# pkill -f llm_pretrain_burn 2>/dev/null || true
# sleep 3
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pwd

# cd ./talas_vlm_embed
# bash ./project_commands.sh


# cd ./spectral-guided-learning
# bash ./project_commands.sh
