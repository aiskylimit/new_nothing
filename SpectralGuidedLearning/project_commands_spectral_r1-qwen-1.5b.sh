#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

[[ -f data/r1-qwen-1.5b/train-segmented.jsonl ]]     || bash scripts/data/data_r1-qwen-1.5b.sh
[[ -f data/r1-qwen-1.5b/spectral-strengths.parquet ]] || bash scripts/capture/capture_r1-qwen-1.5b.sh
[[ -f data/r1-qwen-1.5b/train-spectral.jsonl ]]      || bash scripts/masks/masks_r1-qwen-1.5b.sh

bash scripts/spectral/spectral_lora_r1-qwen-1.5b.sh

bash scripts/spectral/spectral_full_r1-qwen-1.5b.sh

deactivate 2>/dev/null || true
unset VIRTUAL_ENV

bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/spectral-lora-r1-qwen-1.5b spectral-lora-r1-qwen-1.5b
bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/spectral-full-r1-qwen-1.5b spectral-full-r1-qwen-1.5b

"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" -m sgl.eval.compare
