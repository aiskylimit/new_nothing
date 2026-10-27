#1 +10
#uni3
#v1

# cd SpectralGuidedLearning && GPUS=5 bash project_commands_b200_gain.sh 

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-iepenED/logs.tar.gz +a

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 2
nvidia-smi

# ls OPSD/results/raw/olmo3-7b-think/grpo/

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./multi-mode-distill
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh

# cd ./cypher-extract
# bash ./project_command.sh

cd ./unified-ie
# tar -czf logs.tar.gz logs/
# ls -lh logs.tar.gz
# bash ./run2.sh
bash ./run3.sh
# bash ./project_commands.sh

# cd ./eval-impact
# CUDA_VISIBLE_DEVICES=2,3 bash ./project_commands.sh

# cd VLM_Distillation-main
# bash ./project_commands.sh
# bash ./project_commands_collect_eval_summary.sh
# bash ./project_commands_eval_base_model_1.sh
# pwd && ls
