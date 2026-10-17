#!/usr/bin/env bash
# Answer-only SFT driver -- DeepSeek-R1-Distill-Qwen-1.5B, FULL fine-tuning.
#   data (s1K-1.1 ground-truth `solution` only, all-ones mask) -> full-FT SFT -> eval -> compare.
# The baseline that isolates the long CoT: same corpus, same prompts and the same hyperparameters
# as project_commands_r1-qwen-1.5b.sh, but the target carries no model-generated reasoning at all,
# just the source's own solution ending in \boxed{}. No capture/spectral phase is needed.
# Every phase runs in ONE env: spectral_guided_learning (../spectral_guided_learning.txt).
# Eval follows P-ALIGN/src/test.py: thinking OFF, n=3, T=0.6, top_p=0.9, repetition_penalty=1.05,
# max_tokens=4096 over AIME24/AIME25/AMC12/MATH500 (Pass@1 + Pass@3).
#   GPUS=0 bash project_commands_answer_r1-qwen-1.5b.sh
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Which GPU(s) each phase runs on (space-separated ids). Training runs torchrun over the whole
# list (effective batch fixed at 32, so 1/2/4/8 GPUs); eval uses the whole list too.
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

# Thinking is OFF here, unlike the long-CoT driver: this arm has no reasoning to supervise, so
# R1-Distill's force-opened <think> block is closed in the prompt and the target is the bare
# ground-truth solution. That is also the prompt the eval below renders, so train and eval see
# the same format. (ENABLE_THINKING=true would instead supervise "</think>\n\n" + solution
# inside the open block -- still answer-only, but a different prompt format than at eval.)
export ENABLE_THINKING="${ENABLE_THINKING:-false}"

# ============================ TRAIN ============================
# data_answer.sh emits train-segmented.jsonl + the all-ones train-vanilla.jsonl in one step.
bash scripts/data/data_answer.sh r1-qwen-1.5b
bash scripts/sft/sft_answer_r1-qwen-1.5b.sh

# ============================ EVAL =============================
# Start from a clean shell env so the eval script activates the vLLM env. Thinking is OFF at eval
# regardless of how the data was built -- override with ENABLE_THINKING_EVAL=true.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
ENABLE_THINKING="${ENABLE_THINKING_EVAL:-false}" \
  bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/answer-r1-qwen-1.5b answer-r1-qwen-1.5b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" -m sgl.eval.compare
