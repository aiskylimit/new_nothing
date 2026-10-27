#!/usr/bin/env bash
# Create conda env "impact" and install IMPACT dependencies.
# Usage (from repo root):
#   bash install_impact.sh
#   conda activate impact
#   python -m spacy download en_core_web_sm
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${IMPACT_CONDA_ENV:-impact}"
PYTHON_VERSION="${IMPACT_PYTHON_VERSION:-3.13}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; activating and updating packages."
  conda activate "${ENV_NAME}"
else
  echo "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  conda activate "${ENV_NAME}"
fi

echo "Installing CUDA nvcc (required by DeepSpeed)..."
conda install -y -c nvidia cuda-nvcc=12.4

echo "Installing Python packages from requirements.txt..."
pip install -r "${REPO_ROOT}/requirements.txt"

# ALM: vendored tokenkit at ALM/tokenkit-main (not PyPI tokenkit)
if [[ -d "${REPO_ROOT}/ALM/tokenkit-main" ]]; then
  if pip show tokenkit >/dev/null 2>&1; then
    _tk="$(python -c "import tokenkit; print(getattr(tokenkit,'__file__',''))" 2>/dev/null || true)"
    if [[ -n "${_tk}" && "${_tk}" != *"${REPO_ROOT}/ALM/tokenkit-main"* ]]; then
      pip uninstall -y tokenkit
    fi
  fi
  pip install -e "${REPO_ROOT}/ALM/tokenkit-main" --no-deps
fi

echo ""
echo "Done. Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_ROOT}"
echo "  python -m spacy download en_core_web_sm   # ResidualKD span matching"
echo "  # Place weights in model_hub/ and data in data/dolly/"
echo "  bash scripts/gpt2_120m/ALM_IMPACT.sh"
