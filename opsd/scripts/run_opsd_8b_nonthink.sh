#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STUDENT_THINKING=False
export TEACHER_THINKING=False
export OPSD_CLIP_8B="${OPSD_CLIP_8B:-1e-7}"
exec bash "${SCRIPT_DIR}/run_training.sh" opsd 8b
