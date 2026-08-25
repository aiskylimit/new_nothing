#d
#datasets
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip /mnt/local/_data/vlm2vec/data
#models
--hf Qwen/Qwen3-14B /mnt/local/_models/@PROJECT@/Qwen3-14B
#segd
#v2


## wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
## sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh

nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# sleep 5
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



# cd ./talas_vlm_embed
# bash ./project_commands.sh
