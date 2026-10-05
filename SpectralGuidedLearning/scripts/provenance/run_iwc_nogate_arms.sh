#!/usr/bin/env bash
# IWC-Stable WITHOUT the spectral gate (energy threshold p=1.0): every response token is supervised
# exactly as in P-ALIGN / sft-nll; the only change is the entropy-based step weights (tau 2.0, clip 2.0).
#   iwc-nogate     : lambda 1.0
#   iwc-nogate-l05 : lambda 0.5
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE}"
export GPUS="${GPUS:-0}" TRACK="${TRACK:-r1-qwen-1.5b-palign}"
for pair in "iwc-nogate:iwc-stable-nogate" "iwc-nogate-l05:iwc-stable-nogate-l05"; do
  ARM="${pair%%:*}"; VARIANT="${pair##*:}"
  echo "===================== ARM ${ARM} (${VARIANT}) ====================="
  bash scripts/provenance/train_arm_r1-qwen-1.5b.sh "${ARM}" "${VARIANT}" || { echo "ARM ${ARM}: TRAINING FAILED" >&2; continue; }
  ( unset VIRTUAL_ENV; ENABLE_THINKING=false \
      bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${ARM}-${TRACK}" "${ARM}-${TRACK}" ) || echo "ARM ${ARM}: EVAL FAILED" >&2
done
