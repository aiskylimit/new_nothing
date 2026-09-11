#!/usr/bin/env bash
# Phase 6: eval a Qwen3-8B LoRA adapter with vLLM, Pass@1 + Pass@3. Default = spectral ckpt;
# pass a checkpoint path + tag to eval another (e.g. the vanilla baseline).
#   ./scripts/eval/eval_qwen3-8b.sh
#   ./scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-qwen3-8b vanilla-qwen3-8b
set -euo pipefail

read -ra GPUS <<< "${GPUS:-6 7}"
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
MODEL="${BASE_PATH}/checkpoints/spectral-qwen3-8b"
BASE_MODEL="${LOCAL_MODELS_ROOT}/Qwen3-8B"
TAG="spectral-qwen3-8b"
[[ -n "${1:-}" ]] && MODEL="$1"
[[ -n "${2:-}" ]] && TAG="$2"
BENCHMARKS="math500,aime24,aime25,amc12"
TEMPERATURE=0.6
TOP_P=0.9
REP_PENALTY=1.05      # P-ALIGN/scripts/Inference.sh
N_SAMPLES=3           # P-ALIGN reports Pass@1 and Pass@3
MAX_TOKENS=30720
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.9
SEED=42
CHAT_TEMPLATE=true
ENABLE_THINKING=false  # P-ALIGN/src/test.py evaluates with enable_thinking=False
ENFORCE_EAGER=true
LORA_R=16
RESULTS_DIR="${BASE_PATH}/results"

OPTS=""
OPTS+=" --model ${MODEL}"
OPTS+=" --tag ${TAG}"
OPTS+=" --benchmarks ${BENCHMARKS}"
OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --top-p ${TOP_P}"
OPTS+=" --repetition-penalty ${REP_PENALTY}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
[[ "${ENFORCE_EAGER}" == true ]] && OPTS+=" --enforce-eager" || OPTS+=" --no-enforce-eager"
OPTS+=" --seed ${SEED}"
[[ "${CHAT_TEMPLATE}" == true ]] && OPTS+=" --chat-template" || OPTS+=" --no-chat-template"
[[ "${ENABLE_THINKING}" == true ]] && OPTS+=" --enable-thinking" || OPTS+=" --no-enable-thinking"
# Checkpoint type is inferred from its actual contents: every train script currently saves
# --no-lora-merge (adapter-only), which writes adapter_config.json instead of a full model config.
OPTS+=" --base-model ${BASE_MODEL}"
if [[ -f "${MODEL}/adapter_config.json" ]]; then
  OPTS+=" --lora-adapter"
  OPTS+=" --lora-r ${LORA_R}"
else
  OPTS+=" --no-lora-adapter"
fi
OPTS+=" --results-dir ${RESULTS_DIR}"

CMD="python ${BASE_PATH}/src/evaluate.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
