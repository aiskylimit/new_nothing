#!/usr/bin/env bash
set -e

source /mnt/local/uvenvs/vlm-distill-eval/bin/activate

export CUDA_VISIBLE_DEVICES=6
bash scripts/eval/run_one_baseline_model_fastvlm_05b.sh
bash scripts/eval/run_one_baseline_model_qwen2_vl_2b.sh
bash scripts/eval/run_one_baseline_model_qwen25_vl_7b.sh
bash scripts/eval/collect_base_model_summaries.sh
