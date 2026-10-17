#!/usr/bin/env bash
# Provenance-aware SFT on P-ALIGN data -- DeepSeek-R1-Distill-Qwen-1.5B, FULL fine-tuning.
# P-ALIGN targets = teacher (R1) prefix + student-generated continuation, trained by P-ALIGN with one
# uniform NLL. Every arm below trains on the SAME tokens (all-ones mask = P-ALIGN's supervision) with
# the IWC / spectral-full recipe; only the per-token objective differs:
#   prov-nll-dft : NLL on the teacher prefix, DFT (-sg(p) log p) on the continuation   <- proposed
#   sft-nll      : NLL everywhere  (P-ALIGN reproduced through this pipeline)
#   sft-dft      : DFT everywhere  (arXiv 2508.05629)
#   prov-dft-nll : reversed assignment (control: is it the provenance split or just "some DFT"?)
# Needs data/r1-qwen-1.5b-palign/train-{segmented,vanilla}.jsonl (project_commands_iwc_palign_r1-qwen-1.5b.sh).
#   GPUS=0 bash project_commands_provenance_r1-qwen-1.5b.sh            # all arms
#   GPUS=0 bash project_commands_provenance_r1-qwen-1.5b.sh sft-dft    # selected arms
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"
export TRACK="${TRACK:-r1-qwen-1.5b-palign}"
export PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE}/.." && pwd)/iwc}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

declare -A ARM_ARGS=(
  [prov-nll-dft]="provenance --prefix-objective nll --cont-objective dft"
  [sft-nll]="vanilla --objective nll"
  [sft-dft]="vanilla --objective dft"
  [prov-dft-nll]="provenance --prefix-objective dft --cont-objective nll"
)
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || ARMS=(prov-nll-dft sft-nll sft-dft prov-dft-nll)

[[ -f "data/${TRACK}/train-provenance.jsonl" ]] || \
  PYTHONPATH="${BASE}/src" "${PROJECT_ENV}/bin/python" -m sgl.transforms.provenance \
    --data-dir "data/${TRACK}" --tokenizer "${MODEL_NAME}" || exit 1

for arm in "${ARMS[@]}"; do
  [[ -n "${ARM_ARGS[${arm}]:-}" ]] || { echo "unknown arm ${arm}" >&2; exit 2; }
  echo "===================== ARM ${arm}: ${ARM_ARGS[${arm}]} ====================="
  # shellcheck disable=SC2086
  if ! bash scripts/provenance/train_arm_r1-qwen-1.5b.sh "${arm}" ${ARM_ARGS[${arm}]}; then
    echo "ARM ${arm}: TRAINING FAILED -- skipping its eval" >&2
    continue
  fi
  # Same eval as every other row: thinking off, P-ALIGN prompt, 4096 tokens, n=3.
  ( unset VIRTUAL_ENV; ENABLE_THINKING=false \
      bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${arm}-${TRACK}" "${arm}-${TRACK}" ) \
    || echo "ARM ${arm}: EVAL FAILED" >&2
done

"${PROJECT_ENV}/bin/python" -m sgl.eval.compare
