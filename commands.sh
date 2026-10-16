#d
#models
--hf Qwen/Qwen2.5-3B-Instruct /mnt/local/@PROJECT@/grit/models/Qwen2.5-3B-Instruct
--hf Qwen/Qwen3Guard-Gen-4B /mnt/local/@PROJECT@/grit/models/Qwen3Guard-Gen-4B
--hf allenai/wildguard /mnt/local/@PROJECT@/grit/models/wildguard
#datasets
--hf-dataset fushinguyenex/GRIT-artifacts /mnt/local/@PROJECT@/grit/prepared
--hf-dataset cais/mmlu /mnt/local/@PROJECT@/grit/data/raw/mmlu
--hf-dataset m-a-p/SuperGPQA /mnt/local/@PROJECT@/grit/data/raw/SuperGPQA
--hf-dataset openai/gsm8k /mnt/local/@PROJECT@/grit/data/raw/gsm8k
--hf-dataset HuggingFaceH4/MATH-500 /mnt/local/@PROJECT@/grit/data/raw/MATH-500
--hf-dataset Hothan/OlympiadBench /mnt/local/@PROJECT@/grit/data/raw/OlympiadBench
--url https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test.jsonl /mnt/local/@PROJECT@/grit/data/raw/livecodebench
--url https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json /mnt/local/@PROJECT@/grit/data/raw/alpaca_eval
--hf-dataset Paul/XSTest /mnt/local/@PROJECT@/grit/data/raw/XSTest
--url https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv /mnt/local/@PROJECT@/grit/data/raw/or-bench
--url https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv /mnt/local/@PROJECT@/grit/data/raw/advbench
--url https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv /mnt/local/@PROJECT@/grit/data/raw/harmbench
--url https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv /mnt/local/@PROJECT@/grit/data/raw/JBB-Behaviors
--hf-dataset sorry-bench/sorry-bench-202406 /mnt/local/@PROJECT@/grit/data/raw/sorry-bench-202406
--url https://huggingface.co/datasets/declare-lab/HarmfulQA/resolve/main/data_for_hub.json /mnt/local/@PROJECT@/grit/data/raw/HarmfulQA
--url https://huggingface.co/datasets/Babelscape/ALERT/resolve/main/alert.jsonl /mnt/local/@PROJECT@/grit/data/raw/ALERT

#glog
#v2

# cd SpectralGuidedLearning && GPUS=5 bash project_commands_b200_gain.sh 

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/pl_logs/
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/collected_logs.tar.gz +a

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

cd ./GRIT
bash ./project_commands.sh
# cd ./cypher-extract
# bash ./project_command.sh

# cd ./OpenED
# bash gather_pl_logs.sh 
# bash gather_logs.sh all35
# tar -czf collected_logs.tar.gz collected_logs/all35/
# ls -lh collected_logs.tar.gz
# bash ./project_commands_19.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./offline_olmo7b_b200
# bash ./project_commands.sh
