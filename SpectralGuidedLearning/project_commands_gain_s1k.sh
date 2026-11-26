#!/usr/bin/env bash
# Answer-gain experiment driver on s1K-1.1 DeepSeek-R1 traces -- TRAIN everything, then EVAL, then compare.
# Models: r1-qwen-1.5b (full fine-tuning), qwen25-7b and qwen3-8b (LoRA r16). Configs: configs/sgl/<model>-s1k/.
# Arm: IWC-Stable weights from answer gain, lambda 0.5, no spectral gate.
# Steps: split at every blank line ("\n\n", segmenter paragraph).
# Comment out any line you don't want to run.
#   bash project_commands_gain_s1k.sh               # GPU 5
#   GPUS=3 bash project_commands_gain_s1k.sh        # another GPU (space-separated ids for several)
#   bash project_commands_gain_s1k.sh --dry-run     # print every stage command, run nothing
# A rerun skips stages whose output exists (--force redoes them). Extra arguments go to every `sgl run`.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

export GPUS="${GPUS:-5}"

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1
if [[ "${DRY_RUN}" == 0 ]]; then
  source "${PROJECT_ENV}/bin/activate"
  # all local inputs up front, so a missing one does not surface hours into the run
  for path in "${LOCAL_MODELS_ROOT}"/{DeepSeek-R1-Distill-Qwen-1.5B,Qwen2.5-7B-Instruct,Qwen3-8B} "${DATASET_NAME}" "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
    [[ -d "${path}" ]] || { echo "missing ${path} (see download.txt)" >&2; exit 1; }
  done
fi

TRAIN_STAGES=prepare,capture,answer_gain,gain_signal,weights,answer_stats,train

# sgl_run <config> <model dir name> <stages> [extra args]
sgl_run() {
  python -m sgl.cli run "$1" --stages "$3" "run.gpus=[${GPUS// /,}]" \
    "model.name=${LOCAL_MODELS_ROOT}/$2" "dataset=${DATASET_NAME}" "${@:4}"
}

# ============================ TRAIN ============================
# per model: data -> capture -> answer gain -> gain signal -> weights -> answer-step statistics -> train
# statistics and the HTML view of the answer steps: data/<model>-s1k/answer-steps/
sgl_run r1-qwen-1.5b-s1k/iwc-gain-l05   DeepSeek-R1-Distill-Qwen-1.5B "${TRAIN_STAGES}" "$@"
sgl_run qwen25-7b-s1k/iwc-gain-l05-lora Qwen2.5-7B-Instruct           "${TRAIN_STAGES}" "$@"
sgl_run qwen3-8b-s1k/iwc-gain-l05-lora  Qwen3-8B                      "${TRAIN_STAGES}" "$@"

# ============================ EVAL =============================
sgl_run r1-qwen-1.5b-s1k/iwc-gain-l05   DeepSeek-R1-Distill-Qwen-1.5B eval "$@"
sgl_run qwen25-7b-s1k/iwc-gain-l05-lora Qwen2.5-7B-Instruct           eval "$@"
sgl_run qwen3-8b-s1k/iwc-gain-l05-lora  Qwen3-8B                      eval "$@"

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.eval.compare
