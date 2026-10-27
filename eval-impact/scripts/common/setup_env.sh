# Portable env for IMPACT training scripts.
# Source after SCRIPT_DIR is set:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../common/setup_env.sh"

if [[ -z "${SCRIPT_DIR:-}" ]]; then
  echo "setup_env.sh: set SCRIPT_DIR before sourcing (see scripts/gpt2_120m/*.sh)" >&2
  exit 1
fi

if [[ -z "${BASE_PATH:-}" ]]; then
  BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
export BASE_PATH
export IMPACT_ROOT="${BASE_PATH}"

# DeepSpeed imports nvcc from CUDA_HOME; prefer conda nvcc when available.
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  :
elif command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
else
  echo "[IMPACT] nvcc not found. Activate conda env and install CUDA toolkit for DeepSpeed:" >&2
  echo "[IMPACT]   conda activate impact" >&2
  echo "[IMPACT]   conda install -y -c nvidia cuda-nvcc=12.4" >&2
  echo "[IMPACT] Or from repo root:  bash install_impact.sh" >&2
  exit 1
fi

export NCCL_DEBUG="${NCCL_DEBUG:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-True}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

export PYTHONPATH="${BASE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Optional overrides:
#   export IMPACT_CUDA_DEVICES=0,1
#   export IMPACT_EVAL_GPU=0
if [[ -n "${IMPACT_CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${IMPACT_CUDA_DEVICES}"
fi
