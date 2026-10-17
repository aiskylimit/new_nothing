#!/usr/bin/env bash
# Phase 6: eval a DeepSeek-R1-Distill-Qwen-1.5B checkpoint with vLLM, following P-ALIGN's
# eval protocol (P-ALIGN/src/test.py + its project_commands.sh cmd_eval): AIME25/AIME24/AMC12/
# MATH500, chat template with thinking OFF, n=3, T=0.6, top_p=0.9, repetition_penalty=1.05,
# max_tokens=4096 -> Pass@1 and Pass@3.
# Default = the SFT Long-CoT (vanilla) checkpoint; pass a path + tag to eval another arm.
#   ./scripts/eval/eval_r1-qwen-1.5b.sh
#   ./scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/spectral-r1-qwen-1.5b spectral-r1-qwen-1.5b
#   ENABLE_THINKING=true MAX_TOKENS=32768 ./scripts/eval/eval_r1-qwen-1.5b.sh   # thinking-mode eval
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Empty BENCH_DATA_ROOT -> benchmarks.py loads HF repo ids (HF cache); set it to use local mirrors.
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE_PATH}/.." && pwd)/iwc}"
# Always switch to the eval env: the driver may have left the unsloth train env active, which has no vLLM.
if [[ "${VIRTUAL_ENV:-}" != "${PROJECT_ENV}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: eval env not found at ${PROJECT_ENV}; build it from spectral_guided_learning.txt (repo root) or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

MODEL="${BASE_PATH}/checkpoints/vanilla-r1-qwen-1.5b"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TAG="vanilla-r1-qwen-1.5b"
[[ -n "${1:-}" ]] && MODEL="$1"
[[ -n "${2:-}" ]] && TAG="$2"
BENCHMARKS="${BENCHMARKS:-math500,aime24,aime25,amc12}"
# P-ALIGN test.py sampling, verbatim.
TEMPERATURE=0.6
TOP_P=0.9
REPETITION_PENALTY=1.05
N_SAMPLES=3
# One window for prompt + output, as test.py builds it (max_model_len = max_tokens). Two sizes
# only: 4096 for the non-thinking eval (no trace to fit), 32768 for a thinking one -- set
# MAX_TOKENS=32768 and max_model_len follows, so the two can never drift apart.
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_TOKENS}"
# Problems per generate() call; finished ones are written after each batch, so a stop keeps them.
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"   # P-ALIGN test.py default; lower it on a shared GPU
SEED="${EVAL_SEED:-42}"   # vLLM sampling seed
CHAT_TEMPLATE=true
# Thinking OFF, as in test.py. R1-Distill's template ignores enable_thinking and hard-codes an
# open <think>; --palign-prompt closes it exactly as test.py apply_chat() does
# ("<think>\n\n</think>\n\n") and passes token ids without a second BOS -- without closing
# it the model would still emit a full thinking trace. For a thinking eval set ENABLE_THINKING=true
# AND MAX_TOKENS=32768; 4096 would cut the trace off mid-way.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
ENFORCE_EAGER=true
# P-ALIGN/src/evaluation.py: math_verify only (the "palign" grader also ORs oat_math_grader).
GRADER=math_verify
LORA_R=16
RESULTS_DIR="${RESULTS_DIR:-${BASE_PATH}/results}"

OPTS=""
OPTS+=" --model ${MODEL}"
OPTS+=" --tag ${TAG}"
OPTS+=" --benchmarks ${BENCHMARKS}"
OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --top-p ${TOP_P}"
OPTS+=" --repetition-penalty ${REPETITION_PENALTY}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
[[ "${ENFORCE_EAGER}" == true ]] && OPTS+=" --enforce-eager" || OPTS+=" --no-enforce-eager"
OPTS+=" --seed ${SEED}"
[[ "${CHAT_TEMPLATE}" == true ]] && OPTS+=" --chat-template" || OPTS+=" --no-chat-template"
[[ "${ENABLE_THINKING}" == true ]] && OPTS+=" --enable-thinking" || OPTS+=" --no-enable-thinking"
OPTS+=" --palign-prompt"
OPTS+=" --grader ${GRADER}"
# Always passed: the base model supplies the chat template / tokenizer, and is what a LoRA
# adapter is loaded onto. Checkpoint type comes from the checkpoint itself -- an adapter_config.json
# means a LoRA adapter, anything else (e.g. the full-FT SFT checkpoint) loads directly.
OPTS+=" --base-model ${BASE_MODEL}"
if [[ -f "${MODEL}/adapter_config.json" ]]; then
  OPTS+=" --lora-adapter"
  OPTS+=" --lora-r ${LORA_R}"
else
  OPTS+=" --no-lora-adapter"
fi
OPTS+=" --results-dir ${RESULTS_DIR}"

CMD="python -m sgl.eval.evaluate ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
