#!/usr/bin/env bash
# Shuffled-entropy control, 3 training seeds, P-ALIGN data:
#   iwc-shuf-l05 : shuffled-entropy control of iwc-nogate-l05 (same gate/lambda/tau/clip, entropies permuted per sample)
# Only the training seed changes; eval (seed 42, n=3, T=0.6, 4096 tok) is identical.
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE}"
export GPUS="${GPUS:-0}" TRACK="${TRACK:-r1-qwen-1.5b-palign}"
for seed in 42 43 44; do
  for spec in "iwc-shuf-l05:iwc-stable-shuffled-nogate-l05:"; do
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
