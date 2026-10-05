#!/usr/bin/env bash
# Offline B200 driver: answer-gain allocation (IWC-Stable, lambda 0.5) behind SGL's spectral gate (p=0.95) with
# LoRA on Qwen2.5-7B-Instruct and Qwen3-8B -> eval under the P-ALIGN protocol. The arms are
# configs/sgl/<model>-palign/iwc-gain-gate95-l05-lora.yaml
# unchanged; only where resources come from differs: the venv, every model and every benchmark are read from
# fixed local paths (override with the env vars below), and every network path is switched off, so a missing
# file fails instead of being downloaded.
#   bash project_commands_b200_gain.sh                           # both models, one after the other, GPU 0
#   MODELS=qwen3-8b GPUS=1 bash project_commands_b200_gain.sh    # one model on another GPU
#   EXTRA_EVAL_SEEDS="43 44" bash project_commands_b200_gain.sh  # + the paper's other sampling seeds
#   DRY_RUN=1 bash project_commands_b200_gain.sh                 # print every stage command, run nothing
# Every stage runs from scratch, data included (prepare -> capture -> answer gain -> gain signal -> gated weights
# -> train -> eval), even when its output exists. RESUME=1 instead skips stages whose output exists, to continue
# an interrupted run.
# Scores go to RESULTS_DIR / EVALSEED_RESULTS_DIR (results_b200, results_b200_evalseed): results/ already tracks
# this machine's summary.json for the same run names, which would make the eval stage skip as done.
# GPU memory: answer gain and eval cap vLLM at GAIN_GPU_MEM_UTIL / EVAL_GPU_MEM_UTIL of the card (0.4 / 0.5,
# ~72 / ~90 GB on a B200); LoRA training (ZeRO-2 offload, samples <= 11.8k tokens) has no hard cap.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothing}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
MODELS="${MODELS:-qwen25-7b qwen3-8b}"
GPUS="${GPUS:-0}"
EXTRA_EVAL_SEEDS="${EXTRA_EVAL_SEEDS:-}"
RESULTS_DIR="${RESULTS_DIR:-results_b200}"
EVALSEED_RESULTS_DIR="${EVALSEED_RESULTS_DIR:-results_b200_evalseed}"
GAIN_GPU_MEM_UTIL="${GAIN_GPU_MEM_UTIL:-0.4}"
EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.5}"

# No network: Hugging Face reads local files only, vLLM sends no usage stats, FlashInfer never fetches cubins
# (it raises instead; install flashinfer-cubin of the same version in the venv if a kernel is missing).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1
export FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

[[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
source "${PROJECT_ENV}/bin/activate"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Fail now rather than hours in if the venv lacks part of the train or eval stack.
python -c "import torch, transformers, peft, deepspeed, vllm, datasets, math_verify, sgl" || {
  echo "${PROJECT_ENV} cannot import the full stack (torch, transformers, peft, deepspeed, vllm, datasets," \
       "math_verify, sgl); point PROJECT_ENV at the venv that has it" >&2
  exit 1
}

model_dir() {
  case "$1" in
    qwen25-7b) echo "${LOCAL_MODELS_ROOT}/Qwen2.5-7B-Instruct" ;;
    qwen3-8b) echo "${LOCAL_MODELS_ROOT}/Qwen3-8B" ;;
    *) echo "unknown model '$1' (qwen25-7b | qwen3-8b)" >&2; return 1 ;;
  esac
}
for key in ${MODELS}; do
  dir="$(model_dir "${key}")"
  [[ -f "${dir}/config.json" ]] || { echo "model not found: ${dir} (set LOCAL_MODELS_ROOT)" >&2; exit 1; }
done
# benchmark dirs as sgl.eval.benchmarks resolves them under BENCH_DATA_ROOT (see download.txt)
for bench in AIME_2024 aime_2025 MATH-500 aimo-validation-amc; do
  [[ -d "${BENCH_DATA_ROOT}/${bench}" ]] || { echo "benchmark not found: ${BENCH_DATA_ROOT}/${bench}" >&2; exit 1; }
done

GPU_LIST="[${GPUS// /,}]"
RUN_FLAGS=()
[[ "${RESUME:-0}" == 1 ]] || RUN_FLAGS+=(--force)
[[ "${DRY_RUN:-0}" == 1 ]] && RUN_FLAGS+=(--dry-run)
for key in ${MODELS}; do
  config="${key}-palign/iwc-gain-gate95-l05-lora"
  overrides=(
    "run.gpus=${GPU_LIST}"
    "model.name=$(model_dir "${key}")"
    "stages.answer_gain.args.gpu-memory-utilization=${GAIN_GPU_MEM_UTIL}"
    "stages.eval.args.gpu-memory-utilization=${EVAL_GPU_MEM_UTIL}"
  )
  echo "===================== ${config} (model $(model_dir "${key}"), GPU ${GPUS}) ====================="
  # prepare -> capture -> answer_gain -> gain_signal -> weights (gate p=0.95) -> train -> eval (sampling seed 42)
  python -m sgl.cli run "${config}" ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} "${overrides[@]}" "results_dir=${RESULTS_DIR}"
  for seed in ${EXTRA_EVAL_SEEDS}; do
    python -m sgl.cli run "${config}" --stages eval ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} "${overrides[@]}" \
      "eval_seed=${seed}" "eval_suffix=-e${seed}" "results_dir=${EVALSEED_RESULTS_DIR}"
  done
done

# <RESULTS_DIR>/comparison-table.md and eval-summary.json from every <RESULTS_DIR>/<tag>/
[[ "${DRY_RUN:-0}" == 1 ]] || python -m sgl.eval.compare --results-dir "${RESULTS_DIR}"
