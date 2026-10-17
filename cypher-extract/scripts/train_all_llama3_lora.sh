#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${LLAMA3_LOG_DIR:-}" ]]; then
  export TRAIN_ALL_LOG_DIR="${LLAMA3_LOG_DIR}"
fi
exec bash "${SCRIPT_DIR}/train_all.sh" llama3 lora "$@"
