#!/usr/bin/env bash
# Phase 6: eval a Qwen2.5-7B-Instruct LoRA adapter with vLLM, Pass@1 + Pass@3. Default = spectral ckpt;
# pass a checkpoint path + tag to eval another (e.g. the vanilla baseline).
#   ./scripts/eval/eval_qwen25-7b.sh
#   ./scripts/eval/eval_qwen25-7b.sh checkpoints/vanilla-qwen25-7b vanilla-qwen25-7b
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Offline server: benchmarks.py resolves aime24/aime25/math500/amc12 from here (see download.txt).
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-/mnt/local/_data/aiskylimit_new_nothingnew_2}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
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

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL="${BASE_PATH}/checkpoints/spectral-qwen25-7b"
BASE_MODEL="${LOCAL_MODELS_ROOT}/Qwen2.5-7B-Instruct"
TAG="spectral-qwen25-7b"
[[ -n "${1:-}" ]] && MODEL="$1"
[[ -n "${2:-}" ]] && TAG="$2"
BENCHMARKS="${BENCHMARKS:-math500,aime24,amc12}"  # math500=primary; aime25 dropped (same tier as aime24, halves small-set cost)
TEMPERATURE=0.6
TOP_P=0.9
REP_PENALTY=1.05      # P-ALIGN/scripts/Inference.sh
N_SAMPLES=3           # P-ALIGN reports Pass@1 and Pass@3
MAX_TOKENS="${MAX_TOKENS:-30720}"         # kept long on purpose: P-ALIGN itself caps at 4096, but
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # training here uses max-seq-len 32768 (s1K-1.1 long CoT)
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

CMD="python -m sgl.eval.evaluate ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
