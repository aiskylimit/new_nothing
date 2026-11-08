#d
#datasets
--hf-dataset Elriggs/openwebtext-100k /mnt/local/@PROJECT@/contra-kd/data/raw/openwebtext-100k
--hf-dataset MiniLLM/dolly /mnt/local/@PROJECT@/contra-kd/data/raw/dolly
--hf-dataset MiniLLM/self-inst /mnt/local/@PROJECT@/contra-kd/data/raw/self-inst
--hf-dataset MiniLLM/Vicuna /mnt/local/@PROJECT@/contra-kd/data/raw/vicuna
--hf-dataset MiniLLM/sinst /mnt/local/@PROJECT@/contra-kd/data/raw/sinst
#models
--hf bachthetrollface/gpt2-1.5B-teacher-dolly /mnt/local/@PROJECT@/contra-kd/results/gpt2/train/sft/gpt2-xlarge
--hf bachthetrollface/gpt2-120M-init-dolly /mnt/local/@PROJECT@/contra-kd/results/gpt2/train/init/gpt2-base
 
#contra1
#v1


#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Embed_Gvendi/MMEB-evaloutputs-json/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/eval-impact/outputs/llama2-tinyllama-opd/logs/dolly-dskd-opd-10.log

# sleep 2
# kill -9 $(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# for d in datasets vlm2vec_eval vlm2vec_train; do
#   rsync -a --delete VLM_Embed_Gvendi/$d/ cred/$d/
# done

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./cred/multi_gpu_v2.py

# cd ./cred
# bash ./project_commands.sh

# cd ./multi-mode-distill
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh

# cd ./VLM_Embed_Gvendi
# bash ./project_commands_2.sh
# bash ./project_commands.sh

# cd ./unified-ie
# bash ./project_commands.sh

# cd ./eval-impact
# tree outputs
# CUDA_VISIBLE_DEVICES=2 bash ./run_8k_b200.sh


# cd VLM_Distillation-main
# bash ./project_commands.sh
# bash ./project_commands_collect_eval_summary.sh
# bash ./project_commands_eval_base_model_1.sh