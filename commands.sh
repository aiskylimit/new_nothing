#d
#datasets
--hf-dataset TIGER-Lab/MMEB-train /mnt/local/@PROJECT@/cred/vlm2vec_train/MMEB-train
--hf-dataset VoCuc/vlm-teacher-embedding /mnt/local/@PROJECT@/cred/datasets
--hf-dataset TIGER-Lab/MMEB-eval /mnt/local/@PROJECT@/cred/vlm2vec_eval/MMEB-eval
--url https://huggingface.co/VoCuc/en_core_web_sm/resolve/main/en_core_web_sm.zip /mnt/local/@PROJECT@/cred/en_core_web_sm.zip
#models
--hf raghavlite/B3_Qwen2_2B /mnt/local/@PROJECT@/cred/models/B3_Qwen2_2B
--hf raghavlite/B3_Qwen2_7B /mnt/local/@PROJECT@/cred/models/B3_Qwen2_7B
--hf apple/FastVLM-0.5B /mnt/local/@PROJECT@/cred/models/FastVLM-0.5B
--hf apple/FastVLM-1.5B /mnt/local/@PROJECT@/cred/models/FastVLM-1.5B
--hf llava-hf/llava-onevision-qwen2-0.5b-ov-hf /mnt/local/@PROJECT@/cred/models/llava-onevision-qwen2-0.5b-ov-hf
--hf Qwen/Qwen2-VL-2B-Instruct /mnt/local/@PROJECT@/cred/models/Qwen2-VL-2B-Instruct

#log
#v1


#2 -f-/mnt/local/aiskylimit_new_nothing/cred/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/eval-impact/outputs/llama2-tinyllama-opd/logs/dolly-dskd-opd-10.log

# sleep 2
# kill -9 $(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)
# kill -9 117225
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