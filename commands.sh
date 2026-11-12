#d
#datasets
--url https://huggingface.co/datasets/VoCuc/UltraInteract-Infer/resolve/main/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/raw/Qwen/Qwen2.5-14B-Instruct/
--url https://huggingface.co/datasets/VoCuc/UltraInteract-Infer/resolve/main/google/gemma-2-9b-it/generated_train.jsonl /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/raw/google/gemma-2-9b-it/
--hf-dataset openai/gsm8k /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/gsm8k
--hf-dataset qintongli/GSM-Plus /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/gsm_plus
--hf-dataset EleutherAI/hendrycks_math /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/hendrycks_math
--hf-dataset google-research-datasets/mbpp /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/mbpp
--hf-dataset allenai/sciq /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/sciq
--hf-dataset cais/mmlu /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/mmlu
--hf-dataset TIGER-Lab/MMLU-Pro /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/mmlu_pro
--hf-dataset SaylorTwift/bbh /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/bbh
--url https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/code_eval.py /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/code_eval/
--url https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/execute.py /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/eval/code_eval/
#models
--hf Qwen/Qwen2.5-1.5B-Instruct /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/models/Qwen2.5_1.5B-Instruct
--hf Qwen/Qwen2.5-14B-Instruct /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/models/Qwen2.5_14B-Instruct


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