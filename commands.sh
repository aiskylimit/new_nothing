#d
#models
--hf black-forest-labs/FLUX.1-dev /mnt/local/_models/@PROJECT@/FLUX.1-dev
--hf gabeguofanclub/flux-1-dev-flowmap-lsd /mnt/local/_models/@PROJECT@/flux-1-dev-flowmap-lsd
--hf THUDM/ImageReward /mnt/local/_models/@PROJECT@/ImageReward
--hf google-bert/bert-base-uncased /mnt/local/_models/@PROJECT@/bert-base-uncased
--hf Qwen/Qwen2.5-VL-3B-Instruct /mnt/local/_models/@PROJECT@/Qwen2.5-VL-3B-Instruct
--url https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt /mnt/local/_models/@PROJECT@/clip/ViT-B-32.pt
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

# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# sleep 5
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



# cd ./talas_vlm_embed
# bash ./project_commands.sh
# tree / -L 5
