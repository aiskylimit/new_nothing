#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${QWEN_LOG_DIR:-}" ]]; then
  export TRAIN_ALL_LOG_DIR="${QWEN_LOG_DIR}"
fi
exec bash "${SCRIPT_DIR}/train_all.sh" qwen3 full_finetune_normalized "$@"
