#!/usr/bin/env bash
# Rebuttal ablation: train on the answer-containing steps ONLY -- the loss of every other step is masked out.
# Same students, s1K-1.1 data, schedule and eval as project_commands.sh; plain NLL, no answer gain.
#   bash project_commands_answer_steps.sh                       # answer string anywhere in the step, GPU 5
#   ANSWER_DEF=boxed bash project_commands_answer_steps.sh      # only steps that write \boxed{answer}
#   bash project_commands_answer_steps.sh --dry-run             # print every stage command, run nothing
# Run it after project_commands.sh, or at least not at the same time: both write data/<model>-s1k/.
# Outputs: rebuttal/results/ansstep-<definition>[-lora]-<model>-s1k/, rebuttal/logs/.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

export GPUS="${GPUS:-5}"

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothing}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
MODEL_DIRS=(DeepSeek-R1-Distill-Qwen-1.5B Qwen2.5-7B-Instruct Qwen3-8B)
# Models: LOCAL_MODELS_ROOT if set, else the first of the two mirrors that holds all three.
if [[ -z "${LOCAL_MODELS_ROOT:-}" ]]; then
  for root in /mnt/local/_models/aiskylimit_new_nothing /mnt/local/_models/aiskylimit_new_nothingnew_2; do
    LOCAL_MODELS_ROOT="${root}"
    complete=1
    for model in "${MODEL_DIRS[@]}"; do [[ -f "${root}/${model}/config.json" ]] || complete=0; done
    [[ "${complete}" == 1 ]] && break
  done
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1

# ============================ CHECK ============================
if [[ "${DRY_RUN}" == 0 ]]; then
  # the full console output of this launch, kept across reruns (stage logs are overwritten)
  mkdir -p rebuttal/logs
  exec > >(tee -a "rebuttal/logs/driver-answer-steps-$(date +%Y%m%d-%H%M%S).log") 2>&1
  echo "[$(date '+%F %T')] commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), GPUS=${GPUS}, args: $*"
  echo "venv: ${PROJECT_ENV}  models: ${LOCAL_MODELS_ROOT}  data: ${LOCAL_DATA_ROOT}"

  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"

  echo "--- ${LOCAL_DATA_ROOT}"; ls "${LOCAL_DATA_ROOT}" || true
  echo "--- ${LOCAL_MODELS_ROOT}"; ls "${LOCAL_MODELS_ROOT}" || true
  # every missing input is listed before stopping, so one launch shows all of them
  missing=0
  for path in "${DATASET_NAME}" "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
    [[ -d "${path}" ]] || { echo "missing dataset: ${path} (see download.txt)" >&2; missing=1; }
  done
  for model in "${MODEL_DIRS[@]}"; do
    [[ -f "${LOCAL_MODELS_ROOT}/${model}/config.json" ]] || {
      echo "missing model: ${LOCAL_MODELS_ROOT}/${model} (see download.txt)" >&2; missing=1; }
  done
  [[ "${missing}" == 0 ]] || exit 1

  nvidia-smi -i "${GPUS// /,}" --query-gpu=index,name,memory.total,memory.used --format=csv
  # fail now rather than hours in if the venv lacks part of the train or eval stack
  CUDA_VISIBLE_DEVICES="${GPUS// /,}" python -c "
import sys, torch, transformers, peft, deepspeed, vllm, datasets, math_verify, pandas, matplotlib, sgl
print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__,
      'vllm', vllm.__version__)
assert torch.cuda.is_available(), 'CUDA is not available'
"
fi

ANSWER_DEF="${ANSWER_DEF:-mention}"
STAGES=prepare,answer_mask,train

# sgl_run <config> <model dir name> <stages> [extra args]
sgl_run() {
  python -m sgl.cli run "$1" --stages "$3" "run.gpus=[${GPUS// /,}]" "answer_definition=${ANSWER_DEF}" \
    "model.name=${LOCAL_MODELS_ROOT}/$2" "dataset=${DATASET_NAME}" "${@:4}"
}

# ============================ TRAIN ============================
sgl_run r1-qwen-1.5b-s1k/answer-steps-only   DeepSeek-R1-Distill-Qwen-1.5B "${STAGES}" "$@"
sgl_run qwen25-7b-s1k/answer-steps-only-lora Qwen2.5-7B-Instruct           "${STAGES}" "$@"
sgl_run qwen3-8b-s1k/answer-steps-only-lora  Qwen3-8B                      "${STAGES}" "$@"

# ============================ EVAL =============================
sgl_run r1-qwen-1.5b-s1k/answer-steps-only   DeepSeek-R1-Distill-Qwen-1.5B eval "$@"
sgl_run qwen25-7b-s1k/answer-steps-only-lora Qwen2.5-7B-Instruct           eval "$@"
sgl_run qwen3-8b-s1k/answer-steps-only-lora  Qwen3-8B                      eval "$@"

# =========================== COMPARE ==========================
# every run under rebuttal/results, the answer-gain ones included
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.eval.compare --results-dir rebuttal/results
