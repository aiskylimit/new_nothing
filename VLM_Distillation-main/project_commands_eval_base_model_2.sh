#!/usr/bin/env bash
set -e

source /mnt/local/uvenvs/vlm-distill-eval/bin/activate

export CUDA_VISIBLE_DEVICES=7
# bash scripts/eval/run_one_baseline_model_qwen3_vl_4b.sh
# bash scripts/eval/run_one_baseline_model_qwen3_vl_8b.sh
bash scripts/eval/run_one_baseline_model_qwen25_vl_3b.sh
bash scripts/eval/collect_base_model_summaries.sh
