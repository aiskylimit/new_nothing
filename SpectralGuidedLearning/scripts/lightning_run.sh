#!/usr/bin/env bash
# Bootstrap + run the plain-spectral Qwen2.5-7B-Instruct pipeline on a single-GPU Lightning AI
# Studio (tested against a 96GB RTX PRO 6000 Blackwell -- matches this repo's cu130 torch/vllm
# pin in pyproject.toml natively, no index change needed).
#
# Usage: upload/clone this repo onto the Studio, cd into SpectralGuidedLearning/, then:
#   bash scripts/lightning_run.sh
#
# Unlike the /mnt/local offline-server scripts this repo was originally written for, a Lightning
# Studio has normal internet access, so the training set (s1K-1.1) and eval benchmarks
# (math500/aime24/amc12) are pulled straight from the HF Hub -- only the base model is
# downloaded locally, because the train/eval scripts always resolve it under LOCAL_MODELS_ROOT.
set -euo pipefail

STUDIO_ROOT="${STUDIO_ROOT:-/teamspace/studios/this_studio}"
export GPUS="${GPUS:-0}"                                                    # single-GPU studio
export PROJECT_ENV="${PROJECT_ENV:-${STUDIO_ROOT}/uvenvs/spectral_guided_learning}"
export LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-${STUDIO_ROOT}/models}"
export DATASET_NAME="${DATASET_NAME:-simplescaling/s1K-1.1}"                # HF repo id directly
export BENCH_DATA_ROOT=""                                                   # "" (set, not unset) -> evaluate.py also pulls straight from HF Hub

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"
export PYTHONPATH="${BASE_PATH}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 1) uv
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# 2) This Studio's base image refuses to create a new venv (uv venv creation fails no matter
# what UV_CACHE_DIR/UV_LINK_MODE are set to -- see prior commits). Install the deps straight into
# the Studio's existing Python instead of a project venv, skipping scripts/setup.sh entirely.
[[ -n "${HF_TOKEN:-}" ]] || echo "note: HF_TOKEN not set -- fine for this run (no gated datasets used)"
uv pip install --system \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  vllm==0.27.1 transformers==5.5.3 "peft>=0.13" "deepspeed>=0.15" \
  "datasets>=2.20" "accelerate>=1.0" "numpy>=2.0,<2.5" "pandas>=2.2" \
  pyarrow matplotlib pyyaml tqdm math-verify==0.8.0 json-repair pytest

python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
from vllm import LLM
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda} | vllm import OK")
PY

# Every track script (data/capture/masks/spectral/eval) does:
#   [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh ; source "${PROJECT_ENV}/bin/activate"
# Stub that file out as a no-op so those checks pass without re-triggering setup.sh's uv sync
# (which would hit the same venv-creation failure) -- packages are already on the system Python.
mkdir -p "${PROJECT_ENV}/bin"
: > "${PROJECT_ENV}/bin/activate"

# 3) pull the base model once (skips if already present from a previous run)
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path(os.environ["LOCAL_MODELS_ROOT"]) / "Qwen2.5-7B-Instruct"
if not model_dir.exists():
    snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=str(model_dir))
    print(f"downloaded -> {model_dir}")
else:
    print(f"already present -> {model_dir}")
PY

# 4) data -> gradient capture -> masks -> spectral train -> eval (3 benchmarks: math500, aime24, amc12)
bash scripts/data/data_qwen25-7b.sh
bash scripts/capture/capture_qwen25-7b.sh
bash scripts/masks/masks_qwen25-7b.sh
bash scripts/spectral/spectral_qwen25-7b.sh
bash scripts/eval/eval_qwen25-7b.sh checkpoints/spectral-qwen25-7b spectral-qwen25-7b

# 5) comparison table (single-model run -- still writes results/comparison-table.md)
python3 -m sgl.eval.compare
