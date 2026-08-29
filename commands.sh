#1 +60
#talas
#v1


# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 /tmp/llm_pretrain_burn.py &

kill -9 $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
sleep 3
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd ./talas_vlm_embed
bash ./project_commands.sh


# cd ./spectral-guided-learning
# bash ./project_commands.sh

# cd ./reward-guidance-main
# bash ./project_commands.sh