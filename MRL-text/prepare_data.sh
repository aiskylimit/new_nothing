#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

# Offline server: local preparation only, after copying data/raw from outside.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MTEB_CACHE="${MTEB_CACHE:-$PWD/data/.cache/mteb}"
"${PYTHON_BIN:-python}" -m utils.prepare_data "$@"
