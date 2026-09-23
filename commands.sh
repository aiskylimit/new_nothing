#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/time_mem/ +a
#talas-time
#v1

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/_run_log_/_run-2026-09-03_17-01-16-VLM-Distillation.log
#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/eval/ +a

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
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
source /mnt/local/uvenvs/talas-vlm-embed/bin/activate
mkdir -p time_mem/time_mem_2
CUDA_VISIBLE_DEVICES=3 bash ./baseline_scripts/train_distill_mse_sigreg_cls.sh 2>&1 | tee time_mem/time_mem_2/train_distill_mse_sigreg_cls.log &
CUDA_VISIBLE_DEVICES=4 bash ./baseline_scripts/train_distill_rkd_sigreg_cls.sh 2>&1 | tee time_mem/time_mem_2/train_distill_rkd_sigreg_cls.log &
CUDA_VISIBLE_DEVICES=5 bash ./baseline_scripts/train_distill_span_attn_cls.sh 2>&1 | tee time_mem/time_mem_2/train_distill_span_attn_cls.log &
CUDA_VISIBLE_DEVICES=6 bash ./script_full/train_distill_sigreg_cls_time_mem.sh 2>&1 | tee time_mem/time_mem_2/train_distill_sigreg_cls_time_mem.log &

# tree -L 2 training
# bash ./project_commands.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &

# cd ./multi-mode-distill
# bash ./project_commands_opsd_ablation.sh
# bash ./project_commands_ablation.sh
# bash ./project_commands.sh

# cd ./cypher-extract
# bash ./project_command.sh

# cd ./reasoning_velocity_distill
# bash ./project_commands.sh