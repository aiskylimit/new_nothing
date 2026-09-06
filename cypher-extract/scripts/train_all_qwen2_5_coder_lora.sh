#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${QWEN2_5_CODER_LOG_DIR:-}" ]]; then
  export TRAIN_ALL_LOG_DIR="${QWEN2_5_CODER_LOG_DIR}"
fi
exec bash "${SCRIPT_DIR}/train_all.sh" qwen2.5_coder lora "$@"
