#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STUDENT_THINKING=False
export TEACHER_THINKING=False
export OPSD_CLIP_4B="${OPSD_CLIP_4B:-1e-6}"
exec bash "${SCRIPT_DIR}/run_training.sh" opsd 4b
