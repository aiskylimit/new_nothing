#!/usr/bin/env bash
# Rebuttal driver: answer-gain allocation on s1K-1.1 DeepSeek-R1 traces -- check, TRAIN, EVAL, compare.
# Models: r1-qwen-1.5b (full fine-tuning), qwen25-7b and qwen3-8b (LoRA r16). Configs: configs/sgl/<model>-s1k/.
# Arm: IWC-Stable weights from answer gain, lambda 0.5, no spectral gate; steps split by sentence.
# Just run it; it activates the venv, checks inputs and environment, then does everything on GPU 5:
#   cd SpectralGuidedLearning && bash project_commands.sh
#   GPUS=3 bash project_commands.sh                 # another GPU (space-separated ids for several)
#   bash project_commands.sh --dry-run              # print every stage command, check and run nothing
# A rerun skips stages whose output exists (--force redoes them). Extra arguments go to every `sgl run`.
# Comment out any line you don't want to run.
# Outputs under rebuttal/: results/ (scores, comparison table), logs/ (one driver-<time>.log per launch with
# everything printed, plus <run>-<stage>.log of the latest run of each stage), answer-steps/<model>-s1k/
# (statistics and HTML view of the answer steps), train-metrics/<run>.json (loss history).
# Datasets stay in data/, checkpoints in checkpoints/.
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
  exec > >(tee -a "rebuttal/logs/driver-$(date +%Y%m%d-%H%M%S).log") 2>&1
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

TRAIN_STAGES=prepare,capture,answer_gain,gain_signal,weights,answer_stats,train

# sgl_run <config> <model dir name> <stages> [extra args]
sgl_run() {
  python -m sgl.cli run "$1" --stages "$3" "run.gpus=[${GPUS// /,}]" \
    "model.name=${LOCAL_MODELS_ROOT}/$2" "dataset=${DATASET_NAME}" "${@:4}"
}

# ============================ TRAIN ============================
# per model: data -> capture -> answer gain -> gain signal -> weights -> answer-step statistics -> train
sgl_run r1-qwen-1.5b-s1k/iwc-gain-l05   DeepSeek-R1-Distill-Qwen-1.5B "${TRAIN_STAGES}" "$@"
sgl_run qwen25-7b-s1k/iwc-gain-l05-lora Qwen2.5-7B-Instruct           "${TRAIN_STAGES}" "$@"
sgl_run qwen3-8b-s1k/iwc-gain-l05-lora  Qwen3-8B                      "${TRAIN_STAGES}" "$@"

# ============================ EVAL =============================
sgl_run r1-qwen-1.5b-s1k/iwc-gain-l05   DeepSeek-R1-Distill-Qwen-1.5B eval "$@"
sgl_run qwen25-7b-s1k/iwc-gain-l05-lora Qwen2.5-7B-Instruct           eval "$@"
sgl_run qwen3-8b-s1k/iwc-gain-l05-lora  Qwen3-8B                      eval "$@"

# =========================== COMPARE ==========================
# writes rebuttal/results/comparison-table.md and eval-summary.json
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.eval.compare --results-dir rebuttal/results

# ======================= PREVIOUS ROUND ========================
# qwen answer-only SFT arm, run before this one. Kept for reference, not run.
# ============================ TRAIN ============================
# per model: data -> capture (spectral + entropy) -> masks -> four matched SFT arms.
# IWC and IWC-Stable share exactly the spectral-selected token set; only weights differ.
# source /mnt/local/uvenvs/spectral_guided_learning_train/bin/activate
# qwen25-7b/qwen3-8b long-CoT arms đã train + eval xong -- tạm tắt; lượt này chỉ chạy arm answer-only.
# bash scripts/data/data_qwen25-7b.sh
# bash scripts/capture/capture_qwen25-7b.sh
# bash scripts/masks/masks_qwen25-7b.sh
# bash scripts/masks/iwc_qwen25-7b.sh
# bash scripts/spectral/spectral_qwen25-7b.sh
# bash scripts/sft/sft_qwen25-7b.sh
# bash scripts/iwc/train_iwc.sh qwen25-7b iwc
# bash scripts/iwc/train_iwc.sh qwen25-7b iwc-stable
# bash scripts/spectral/spectral_unsloth_qwen25-7b.sh
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc-stable
# bash scripts/sft/sft_unsloth_qwen25-7b.sh
# answer-only SFT arm (ground-truth `solution`, no long CoT): data+masks in one step, no
# capture/spectral needed. Data prep and training (train_sft.py + DeepSpeed) both run in the
# main env (spectral_guided_learning.txt); the unsloth variant is kept commented for reference.
# bash scripts/data/data_answer.sh qwen25-7b
# bash scripts/sft/sft_answer.sh qwen25-7b
# bash scripts/sft/sft_answer_unsloth.sh qwen25-7b

# bash scripts/data/data_qwen3-8b.sh
# bash scripts/capture/capture_qwen3-8b.sh
# bash scripts/masks/masks_qwen3-8b.sh
# bash scripts/masks/iwc_qwen3-8b.sh
# bash scripts/spectral/spectral_qwen3-8b.sh
# bash scripts/sft/sft_qwen3-8b.sh
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc-stable
# bash scripts/spectral/spectral_unsloth_qwen3-8b.sh
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc-stable
# bash scripts/sft/sft_unsloth_qwen3-8b.sh
# bash scripts/data/data_answer.sh qwen3-8b
# bash scripts/sft/sft_answer.sh qwen3-8b
# bash scripts/sft/sft_answer_unsloth.sh qwen3-8b

# ============================ EVAL =============================
# Start from a clean shell env so each eval script activates the vLLM env (spectral-guided-learning).
# deactivate 2>/dev/null || true
# unset VIRTUAL_ENV
# only the answer-only checkpoints this run (the other arms are already in results/)

# bash scripts/eval/eval_qwen25-7b.sh
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/vanilla-qwen25-7b vanilla-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-qwen25-7b iwc-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-qwen25-7b iwc-stable-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-unsloth-qwen25-7b iwc-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-unsloth-qwen25-7b iwc-stable-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/vanilla-unsloth-qwen25-7b vanilla-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/answer-qwen25-7b answer-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/answer-unsloth-qwen25-7b answer-unsloth-qwen25-7b

# bash scripts/eval/eval_qwen3-8b.sh
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-qwen3-8b vanilla-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-qwen3-8b iwc-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-qwen3-8b iwc-stable-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-unsloth-qwen3-8b vanilla-unsloth-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/answer-qwen3-8b answer-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/answer-unsloth-qwen3-8b answer-unsloth-qwen3-8b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
# "${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" -m sgl.eval.compare
