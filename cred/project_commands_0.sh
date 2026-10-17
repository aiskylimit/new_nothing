#!/usr/bin/env bash
set -e


source .venv/bin/activate

# bash prepare.sh



CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 0 1 1 1.0 0.5 256 0.02 &
CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 0 1 1.0 0.5 256 0.05 &

wait

CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_vqa.sh 1 1 1 1 1.0 0.5 256 0.02 &
CUDA_VISIBLE_DEVICES=0 bash script_full/train_distill_sigreg_cls.sh 1 1 0 1 1.0 0.5 256 0.07 &

wait


# =========================
# 9. Copy JSON eval outputs
# =========================

# JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json-v5}"

# python json_filter.py ./MMEB-eval_outputs_v5 "${JSON_FILTER_DESTINATION}" --overwrite
