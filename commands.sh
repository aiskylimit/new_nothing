#1 +120
#pga
#v1


#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Embed_Gvendi/MMEB-evaloutputs-json/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a

# sleep 2
# kill -9 $(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./cred/multi_gpu_v2.py


# cd ./cred
# bash ./project_commands.sh

# cd ./multi-mode-distill
# bash ./eval_rivad.sh
# bash ./project_commands.sh

cd ./VLM_Embed_Gvendi
bash ./project_commands.sh

# cd ./unified-ie
# bash ./project_commands.sh


# cd VLM_Distillation-main
# bash ./project_commands_eval_qwen25vl_7b_gpu7.sh
# bash ./project_commands.sh
# bash ./project_commands_collect_eval_summary.sh
# bash ./project_commands_eval_base_model_1.sh