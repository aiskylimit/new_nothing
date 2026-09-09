#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync
source .venv/bin/activate

RUN_GPUS=3 bash scripts/run_teacher_student.sh \
  --families qwen3 \
  --settings all \
  --student-settings full_finetune,full_finetune_normalized \
  --student-methods sft \
  --phase all
