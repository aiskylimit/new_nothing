#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_VENV="${OPD_VENV:-/mnt/local/uvenvs/impact-opd-b200}"
command -v uv >/dev/null || { echo 'uv is required (same environment manager as the supplied template).' >&2; exit 1; }
if [[ ! -x "$OPD_VENV/bin/python" ]]; then uv venv --python 3.11 --seed "$OPD_VENV"; fi
uv pip install --python "$OPD_VENV/bin/python" 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$OPD_VENV/bin/python" -r "$SCRIPT_DIR/impact-opd-b200.txt"
"$OPD_VENV/bin/python" -m pip check
echo "Environment ready: $OPD_VENV"
