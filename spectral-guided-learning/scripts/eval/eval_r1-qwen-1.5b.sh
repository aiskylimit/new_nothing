#!/usr/bin/env bash
# Phase 6: eval a DeepSeek-R1-Distill-Qwen-1.5B LoRA adapter with vLLM, Pass@1. Default = spectral ckpt;
# pass a checkpoint path + tag to eval another (e.g. the vanilla baseline).
#   ./scripts/eval/eval_r1-qwen-1.5b.sh
#   ./scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/vanilla-r1-qwen-1.5b vanilla-r1-qwen-1.5b
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Offline server: benchmarks.py resolves aime24/aime25/math500/amc12 from here (see download.txt).
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-/mnt/local/_data/aiskylimit_new_nothing}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
MODEL="${BASE_PATH}/checkpoints/spectral-r1-qwen-1.5b"
BASE_MODEL="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
TAG="spectral-r1-qwen-1.5b"
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
ENABLE_THINKING=true
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
# Checkpoint type is inferred from the name: a "lora" in the path or tag is loaded as a LoRA
# adapter on the base model; anything else (e.g. the spectral full-FT checkpoint) loads directly.
if [[ "${MODEL}" == *lora* || "${TAG}" == *lora* ]]; then
  OPTS+=" --base-model ${BASE_MODEL}"
  OPTS+=" --lora-adapter"
  OPTS+=" --lora-r ${LORA_R}"
else
  OPTS+=" --base-model ${BASE_MODEL}"   # for chat template / tokenizer; harmless for full-FT
  OPTS+=" --no-lora-adapter"
fi
OPTS+=" --results-dir ${RESULTS_DIR}"

CMD="python ${BASE_PATH}/src/evaluate.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
