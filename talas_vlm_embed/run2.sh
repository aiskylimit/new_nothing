#!/usr/bin/env bash
set -e


source .venv/bin/activate

# bash prepare.sh



CUDA_VISIBLE_DEVICES=2 bash baseline_scripts/train_distill_span_attn_sigreg_cls.sh


# =========================
# 9. Copy JSON eval outputs
# =========================

# JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json-v5}"

# python json_filter.py ./MMEB-eval_outputs_v5 "${JSON_FILTER_DESTINATION}" --overwrite
