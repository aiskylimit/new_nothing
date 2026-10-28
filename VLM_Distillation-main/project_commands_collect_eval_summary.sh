#!/usr/bin/env bash
set -euo pipefail

source /mnt/local/uvenvs/vlm-distill-eval/bin/activate

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

exec python scripts/eval/collect_eval_summaries.py "$@"
