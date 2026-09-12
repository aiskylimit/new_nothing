#d
#datasets
--url https://huggingface.co/datasets/VoCuc/UltraInteract-Infer/resolve/main/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl /mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill/data/raw/Qwen/Qwen2.5-14B-Instruct/
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

#reasoning_velocity_distill
#v2

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v3/ +a
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



# source /mnt/local/uvenvs/talas-vlm-embed/bin/activate
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./talas_vlm_embed/multi_gpu_v2.py



# cd ./talas_vlm_embed
# bash ./project_commands.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_gpu.py &


cd ./reasoning_velocity_distill
bash ./project_commands.sh

# cd ./cypher-extract
# bash ./project_command.sh