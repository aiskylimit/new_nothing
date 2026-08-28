#!/usr/bin/env bash
# Phase 6: eval spectral LoRA adapter (vLLM, Pass@1) on P-ALIGN's 4 benchmarks -- directly comparable to its Table 1 row.
#   ./scripts/qwen25/eval/eval_qwen25-7b.sh [model_path] [tag]
set -euo pipefail

GPUS=(2 3)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${BASE_PATH}/.venv/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${BASE_PATH}/.venv/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

MODEL="${BASE_PATH}/checkpoints/spectral-qwen25-7b"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
TAG="spectral-qwen25-7b"
[[ -n "${1:-}" ]] && MODEL="$1"
[[ -n "${2:-}" ]] && TAG="$2"
BENCHMARKS="math500,aime24,aime25,amc12"
TEMPERATURE=0.6
TOP_P=0.9
N_SAMPLES=1
MAX_TOKENS=30720
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.9
SEED=42
CHAT_TEMPLATE=true
ENABLE_THINKING=false
ENFORCE_EAGER=true
LORA_R=16
RESULTS_DIR="${BASE_PATH}/results"

OPTS=""
OPTS+=" --model ${MODEL}"
OPTS+=" --tag ${TAG}"
OPTS+=" --benchmarks ${BENCHMARKS}"
OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --top-p ${TOP_P}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
[[ "${ENFORCE_EAGER}" == true ]] && OPTS+=" --enforce-eager" || OPTS+=" --no-enforce-eager"
OPTS+=" --seed ${SEED}"
[[ "${CHAT_TEMPLATE}" == true ]] && OPTS+=" --chat-template" || OPTS+=" --no-chat-template"
[[ "${ENABLE_THINKING}" == true ]] && OPTS+=" --enable-thinking" || OPTS+=" --no-enable-thinking"
OPTS+=" --base-model ${BASE_MODEL}"
OPTS+=" --lora-adapter"
OPTS+=" --lora-r ${LORA_R}"
OPTS+=" --results-dir ${RESULTS_DIR}"

CMD="python ${BASE_PATH}/src/evaluate.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
