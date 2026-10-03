#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${ENV_PATH:-/mnt/local/uvenvs/opsd}"

python3.11 -m venv "${ENV_PATH}"
source "${ENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
