#!/usr/bin/env bash
# Install the full stack (training + eval/vLLM) into .venv via uv, pyproject.toml + uv.lock.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

INSTALL_FLASH_ATTN=false

command -v uv >/dev/null || { echo "ERROR: uv not found" >&2; exit 1; }
# torch/torchvision/torchaudio/vllm are pinned in pyproject.toml against a specific CUDA build
# ([tool.uv.index] "pytorch", currently cu130 for Blackwell/B200) -- change that URL, not this
# script, to target different hardware; uv.lock then needs `uv lock` to re-resolve.
uv sync
VENV_PY="${BASE_PATH}/.venv/bin/python"

[[ "${INSTALL_FLASH_ATTN}" == true ]] && uv pip install --python "${VENV_PY}" flash-attn --no-build-isolation

"${VENV_PY}" - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
from vllm import LLM
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda} | vllm import OK")
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  "${VENV_PY}" -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
  echo "HF login OK"
else
  echo "WARN: HF_TOKEN not set (needed for gated datasets, e.g. GPQA in evaluate.py)"
fi
echo "setup done"
