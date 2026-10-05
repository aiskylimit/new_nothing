#!/usr/bin/env bash
# Extra training seeds for the two leading arms on P-ALIGN data (seed 42 already done):
#   iwc-nogate-l05 : IWC-Stable, no spectral gate, lambda 0.5, tau 2.0, clip 2.0
#   sft-nll        : plain NLL (P-ALIGN reproduced through this pipeline)
# Only the training seed changes; eval (seed 42, n=3, T=0.6, 4096 tok) is identical.
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE}"
export GPUS="${GPUS:-0}" TRACK="${TRACK:-r1-qwen-1.5b-palign}"
for seed in 43 44; do
  for spec in "iwc-nogate-l05:iwc-stable-nogate-l05:" "sft-nll:vanilla:--objective nll"; do
    IFS=: read -r name variant extra <<< "${spec}"
    ARM="${name}-s${seed}"
    echo "===================== ARM ${ARM} ====================="
    # shellcheck disable=SC2086
    SEED="${seed}" bash scripts/provenance/train_arm_r1-qwen-1.5b.sh "${ARM}" "${variant}" ${extra} \
      || { echo "ARM ${ARM}: TRAINING FAILED" >&2; continue; }
    ( unset VIRTUAL_ENV; ENABLE_THINKING=false \
        bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${ARM}-${TRACK}" "${ARM}-${TRACK}" ) || echo "ARM ${ARM}: EVAL FAILED" >&2
  done
done
