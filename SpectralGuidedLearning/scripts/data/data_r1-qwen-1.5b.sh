#!/usr/bin/env bash
# Phase 2 (+4): data prep (s1K-1.1 long CoT) -- DeepSeek-R1-Distill-Qwen-1.5B track.
# Emits train-segmented.jsonl and, right away, the all-ones train-vanilla.jsonl that the plain
# SFT Long-CoT arm trains on, so that arm needs no capture/spectral phase. Running the spectral
# path later (capture -> masks_r1-qwen-1.5b.sh --vanilla) rewrites train-vanilla.jsonl with the
# same all-ones masks over the subset of samples that have spectral strengths.
set -euo pipefail
# Data/checkpoint namespace: r1-qwen-1.5b = s1K-1.1 track; e.g. r1-qwen-1.5b-palign = P-ALIGN data.
TRACK="${TRACK:-r1-qwen-1.5b}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE_PATH}/.." && pwd)/iwc}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "data/${TRACK}"

# Model / dataset default to HF Hub ids: downloaded on first use, then read from the HF cache.
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
# s1K-1.1, same corpus as the qwen25-7b / qwen3-8b tracks (and P-ALIGN): its
# deepseek_thinking_trajectory + deepseek_attempt are a real DeepSeek-R1 long CoT, which is
# exactly the format R1-Distill was distilled into.
DATASET_NAME="${DATASET_NAME:-simplescaling/s1K-1.1}"
OUTPUT_PATH="data/${TRACK}/train-segmented.jsonl"
N_SAMPLES="${N_SAMPLES:-}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
# true (right default here): the long CoT is supervised inside the <think> block R1-Distill's
# template force-opens, i.e. the model's native format. false closes that block in the prompt and
# supervises the same CoT as plain prose -- what the qwen25-7b / qwen3-8b tracks use, because
# those templates have no thinking block to fill. Keep this in step with ENABLE_THINKING in
# scripts/eval/eval_r1-qwen-1.5b.sh.
ENABLE_THINKING="${ENABLE_THINKING:-true}"
# true (with ENABLE_THINKING=false): close <think> as P-ALIGN / eval --palign-prompt do
# ("<think>\n\n</think>\n\n"), so the training prompt tokenizes exactly like the eval prompt.
PALIGN_PROMPT="${PALIGN_PROMPT:-false}"

OPTS=""
OPTS+=" --dataset-name ${DATASET_NAME}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --tokenizer ${MODEL_NAME}"
OPTS+=" --output-path ${OUTPUT_PATH}"
OPTS+=" --chat-template"
[[ "${ENABLE_THINKING}" == true ]] && OPTS+=" --enable-thinking" || OPTS+=" --no-enable-thinking"
[[ "${PALIGN_PROMPT}" == true ]] && OPTS+=" --palign-prompt"
if [[ -n "${N_SAMPLES}" ]]; then
  OPTS+=" --n-samples ${N_SAMPLES}"
fi

CMD="python -m sgl.data.prepare ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/${TRACK}-data.log

# All-ones mask = every response token supervised = plain SFT on the full long CoT.
CMD="python -m sgl.selection.build_masks --data-path ${OUTPUT_PATH} --vanilla-only"
echo "${CMD}"
${CMD} 2>&1 | tee logs/${TRACK}-vanilla-masks.log
