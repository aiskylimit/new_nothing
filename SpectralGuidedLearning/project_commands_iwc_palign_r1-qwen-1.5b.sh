#!/usr/bin/env bash
# IWC on P-ALIGN's data -- DeepSeek-R1-Distill-Qwen-1.5B, FULL fine-tuning:
#   data (P-ALIGN prefix-aligned traces) -> capture (spectral + entropy) -> IWC weights
#   -> full-FT IWC SFT -> eval -> compare.
# Same training set and format as P-ALIGN (references/P-ALIGN/configs/r1_distill_qwen_1.5b_palign_full_sft.yaml):
# references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz, thinking OFF at train and eval (prompt ends with an
# empty <think>\n\n</think>\n\n). The only difference from P-ALIGN is which response tokens are
# supervised (spectral selection) and how they are weighted (IWC-Stable), so
# iwc-stable-r1-qwen-1.5b-palign compares directly against the P-ALIGN row.
# Everything lives under data/r1-qwen-1.5b-palign and checkpoints/*-r1-qwen-1.5b-palign, so the
# s1K-1.1 track (project_commands_iwc_r1-qwen-1.5b.sh) is untouched.
#   GPUS=0 bash project_commands_iwc_palign_r1-qwen-1.5b.sh
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

export TRACK="${TRACK:-r1-qwen-1.5b-palign}"
export DATASET_NAME="${DATASET_NAME:-${BASE}/references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz}"
# Thinking OFF, as P-ALIGN trains (enable_thinking: false) and as eval runs.
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export PALIGN_PROMPT=true   # "<think>\n\n</think>\n\n", identical to the eval prompt

export PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE}/.." && pwd)/iwc}"
source "${PROJECT_ENV}/bin/activate"

# ============================ TRAIN ============================
[[ -f "data/${TRACK}/train-segmented.jsonl" ]]     || bash scripts/data/data_r1-qwen-1.5b.sh
[[ -f "data/${TRACK}/spectral-strengths.parquet" ]] || bash scripts/capture/capture_r1-qwen-1.5b.sh

# IWC-Stable weights (Eq. 13-16): lambda 1.0, tau 2.0, clip 2.0. The build log must print
# mean weighted/selected mass=1.000000 for iwc-stable.
export IWC_INTERPOLATION="${IWC_INTERPOLATION:-1.0}"
export IWC_TEMPERATURE="${IWC_TEMPERATURE:-2.0}"
export IWC_CLIP="${IWC_CLIP:-2.0}"
bash scripts/masks/iwc_r1-qwen-1.5b.sh

# DeepSpeed ZeRO-2 (fp32 master weights) is the default in the train script.
bash scripts/iwc/train_iwc_r1-qwen-1.5b.sh iwc-stable

# ============================ EVAL =============================
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
ENABLE_THINKING=false \
  bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/iwc-stable-${TRACK}" "iwc-stable-${TRACK}"

# =========================== COMPARE ==========================
"${PROJECT_ENV}/bin/python" -m sgl.eval.compare
