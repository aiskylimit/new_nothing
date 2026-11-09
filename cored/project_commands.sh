#!/usr/bin/env bash
# Pipeline CORED tren 3 GPU B200; moi du lieu/model deu local.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -n "${CORED_VENV:-}" ]]; then
  source "$CORED_VENV/bin/activate"
elif [[ -f /mnt/local/uvenvs/cored/bin/activate ]]; then
  source /mnt/local/uvenvs/cored/bin/activate
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "${PY:-python}" -m train.pipeline_b200 "$@"
