#d
#models
--hf meta-llama/Llama-2-7b-hf /mnt/local/@PROJECT@/impact_opd/models/Llama-2-7b-hf
--hf TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T /mnt/local/@PROJECT@/impact_opd/models/TinyLlama-1.1B-intermediate-step-1431k-3T
#datasets
--url https://huggingface.co/datasets/pkcii/eval-impact-dolly/resolve/54795e8ce55e166971eeec88c283e66a5e228241/original/train.jsonl /mnt/local/@PROJECT@/impact_opd/raw/dolly_reference/original
--url https://huggingface.co/datasets/pkcii/eval-impact-dolly/resolve/54795e8ce55e166971eeec88c283e66a5e228241/original/dev.jsonl /mnt/local/@PROJECT@/impact_opd/raw/dolly_reference/original
--url https://huggingface.co/datasets/pkcii/eval-impact-dolly/resolve/54795e8ce55e166971eeec88c283e66a5e228241/original/valid.jsonl /mnt/local/@PROJECT@/impact_opd/raw/dolly_reference/original
--url https://huggingface.co/datasets/pkcii/eval-impact-dolly/resolve/54795e8ce55e166971eeec88c283e66a5e228241/manifest.json /mnt/local/@PROJECT@/impact_opd/raw/dolly_reference

#uni
#v2

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
# bash pack_checkpoints.sh
tree results
# tar -czf logs.tar.gz logs/
# ls -lh logs.tar.gz
# bash ./run2.sh
# bash ./project_commands.sh

# cd ./opsd
# bash ./project_commands.sh

# cd VLM_Distillation-main
# bash ./project_commands.sh
# bash ./project_commands_collect_eval_summary.sh
# bash ./project_commands_eval_base_model_1.sh
# pwd && ls
