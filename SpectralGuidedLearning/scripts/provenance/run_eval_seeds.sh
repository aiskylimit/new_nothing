#!/usr/bin/env bash
# Sampling-seed variance: re-evaluate the training-seed-42 checkpoint of each key arm with vLLM
# sampling seeds 43 and 44 (seed 42 is the existing result). Same protocol otherwise: thinking off,
# P-ALIGN prompt, n=3, T=0.6, top_p 0.9, 4096 tokens. Results: results_evalseed/<arm>-e<seed>/.
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE}"
unset VIRTUAL_ENV
TRACK=r1-qwen-1.5b-palign
for eseed in 43 44; do
  for ckpt in sft-nll iwc-nogate-l05 iwc-shuf-l05-s42 iwc-gain-l05-s42; do
    echo "===================== EVAL ${ckpt} sampling-seed ${eseed} ====================="
    GPUS=0 ENABLE_THINKING=false EVAL_SEED="${eseed}" RESULTS_DIR="${BASE}/results_evalseed" \
      bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${ckpt}-${TRACK}" "${ckpt}-e${eseed}" \
      || echo "EVAL ${ckpt} e${eseed} FAILED" >&2
  done
done
