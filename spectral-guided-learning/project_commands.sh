#!/usr/bin/env bash
# Full pipeline (data -> capture -> masks -> train -> eval) for all 4 tracks, unattended.
# Each track fails fast internally (set -e) but a failed track doesn't stop the other one.
set -uo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED_TRACKS=()

run_track() {
  local name="$1"
  shift
  # Not a direct && / || / if operand -- that would suppress the function's `set -e`.
  ( "$@" )
  local status=$?
  if [[ ${status} -eq 0 ]]; then
    echo ">>> track ${name}: OK"
  else
    echo ">>> track ${name}: FAILED (exit ${status}) -- continuing to the next track" >&2
    FAILED_TRACKS+=("${name}")
  fi
}

track_qwen3_1_7b() {
  set -e
  "${BASE_PATH}/scripts/qwen3/data/data_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/data/capture_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/data/masks_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/weight_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/prune_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/sft/sft_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/prucot_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh"
  "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh" \
    "${BASE_PATH}/checkpoints/vanilla-qwen3-1.7b" vanilla-qwen3-1.7b
  "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-1.7b.sh" \
    "${BASE_PATH}/checkpoints/prucot-qwen3-1.7b" prucot-qwen3-1.7b
}

track_qwen3_4b_instruct() {
  set -e
  "${BASE_PATH}/scripts/qwen3/data/data_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/data/capture_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/data/masks_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/weight_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/prune_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/sft/sft_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/spectral/spectral_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/prucot_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh"
  "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh" \
    "${BASE_PATH}/checkpoints/vanilla-qwen3-4b-instruct" vanilla-qwen3-4b-instruct
  "${BASE_PATH}/scripts/qwen3/eval/eval_qwen3-4b-instruct.sh" \
    "${BASE_PATH}/checkpoints/prucot-qwen3-4b-instruct" prucot-qwen3-4b-instruct
}

run_track qwen3-1.7b track_qwen3_1_7b
run_track qwen3-4b-instruct track_qwen3_4b_instruct

# qwen3-8b and qwen25-7b: out of scope for now.

python "${BASE_PATH}/src/compare_results.py"

if [[ ${#FAILED_TRACKS[@]} -gt 0 ]]; then
  echo ">>> failed tracks: ${FAILED_TRACKS[*]} -- see logs/ above for each track's failing step" >&2
  exit 1
fi
