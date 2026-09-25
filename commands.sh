#2
#hierd
#v2

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/_run_log_/_run-2026-09-03_17-01-16-VLM-Distillation.log
#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/eval/ +a

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
kill -9 498 499 500 501 502 503 504 505
# sleep 2
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

cd ./talas_vlm_embed
bash ./project_commands.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &

# cd ./multi-mode-distill
# bash ./project_commands.sh
# bash ./project_commands_ablation.sh
# bash ./context_truncate_ablation.sh


# cd ./cypher-extract
# bash ./project_command.sh
