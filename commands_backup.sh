#2
#segd
#v2


# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &

kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
sleep 5
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



# cd ./talas_vlm_embed
# bash ./project_commands.sh


cd ./spectral-guided-learning
bash ./project_commands.sh