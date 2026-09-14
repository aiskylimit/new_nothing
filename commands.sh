#1 +10
#test
#v2

ps -o pid,ppid,pgid,sid,user,stat,etime,cmd \
  -p 1141359,1141362,1158168,1158235,1158236,1158237,1158238

pstree -aps 1141359
pstree -aps 1158235

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v1/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/_run_log_/_run-2026-09-03_17-01-16-VLM-Distillation.log
#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/eval/ +a

# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=6,7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 3
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
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
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &

# cd ./multi-mode-distill
# bash ./project_commands.sh

# cd ./cypher-extract
# bash ./project_command.sh

# cd ./reasoning_velocity_distill
# bash ./project_commands.sh