#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=false

task_configurations=(
    "aime24 3"
    "amc23 3"
    "math500 3"
    "minerva 3"
    "gpqa 3"
    "olympiad 3"
)

# =========================
# Configuration
# =========================
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_ROOT="outputs_origin"
SEED=0
MAX_TOKENS=4096
TEMPERATURE=0.6
TOP_P=0.9
REPETITION_PENALTY=1.05
GPU_MEM_UTIL=0.8
MAX_MODEL_LEN=4096

for config in "${task_configurations[@]}"; do
  read -r task sample_n <<< "$config"

  echo "=============================================="
  echo "Running task: ${task}"
  echo "Samples per query: ${sample_n}"
  echo "=============================================="

  python -u math_eval.py \
      --model_name_or_path "${MODEL_PATH}" \
      --data_name "${task}" \
      --output_dir "${OUTPUT_ROOT}/${task}/r1_qwen_7b" \
      --split "test" \
      --prompt_type "deepseek-longcot" \
      --num_test_sample -1 \
      --max_tokens_per_call "${MAX_TOKENS}" \
      --seed "${SEED}" \
      --temperature "${TEMPERATURE}" \
      --n_sampling "${sample_n}" \
      --top_p "${TOP_P}" \
      --repetition_penalty "${REPETITION_PENALTY}" \
      --start 0 \
      --end -1 \
      --use_vllm \
      --save_outputs \
      --apply_chat_template \
      --gpu_memory_utilization "${GPU_MEM_UTIL}" \
      --max_model_len "${MAX_MODEL_LEN}"
done
