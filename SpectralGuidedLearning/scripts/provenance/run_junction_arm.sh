#!/usr/bin/env bash
# prov-nll-dft-j8: NLL on the teacher prefix AND on the first 8 continuation tokens (junction),
# DFT on the rest of the continuation. Fix for prov-nll-dft, which looped on "<End_of_Prefix>"
# because DFT gives the near-impossible first post-marker token (p~1e-4) ~no gradient.
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export GPUS="${GPUS:-0}" TRACK="${TRACK:-r1-qwen-1.5b-palign}"
ARM=prov-nll-dft-j8
bash scripts/provenance/train_arm_r1-qwen-1.5b.sh "${ARM}" provenance-j8 \
  --prefix-objective nll --cont-objective dft --junction-objective nll || exit 1
( unset VIRTUAL_ENV; ENABLE_THINKING=false \
    bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${ARM}-${TRACK}" "${ARM}-${TRACK}" )
"${PROJECT_ENV:-$(cd "${BASE}/.." && pwd)/iwc}/bin/python" -m sgl.eval.compare
