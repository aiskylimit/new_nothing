#! /bin/bash
# usage: bash scripts/general_domain/process_data.sh ee_v1
# DATA: folder under data/general_domain (ee_v1, ee_v2, re_tacred, re_fewrel; NER and mt_v2 need --max-prompt-length 480)
DATA=${1:-ee_v1}

export TF_CPP_MIN_LOG_LEVEL=3
MODEL=Qwen/Qwen3-0.6B
[ -d models/Qwen3-0.6B ] && MODEL=models/Qwen3-0.6B   # local copy from download.txt

PYTHONPATH=. python3 ./tools/process_data.py \
    --data-dir ./data/general_domain/${DATA}/ \
    --processed-data-dir ./processed_data/general_domain/${DATA} \
    --model-path ${MODEL} \
    --data-process-workers 4 \
    --max-prompt-length 460 \
    --max-length 768 \
    --dev-num 1000 \
    --model-type qwen
