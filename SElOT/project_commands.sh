#!/bin/bash
# SElOT job on the new_nothing server (offline: no outbound calls; models and source files come from the download job).
# commands.sh runs exactly "cd ./SElOT" + "bash ./project_commands.sh", so every default below must be right as is;
# to change a setting, edit it here. Resubmit the same job after a timeout: every step resumes (see RUN.md).
#
# Every job: rebuild the data from the downloaded source files (fetch_data --src, prepare_eval, build_fit_sets; checked
# against data_manifest.json), then scripts/pipeline.py:
#   label (workflow 1) -> activations at layers 8-22 (sweep 10-22 + AlphaSteer's 8-19) -> SElOT layer sweep on the
#   val split (layers 10-22 = 30-70% of depth, k 5, m 256, alpha 1)
#   -> best layer by mean val DSR -> SElOT full eval -> AlphaSteer full eval -> results/compare_wf1.md
# No W&B. Small outputs: SElOT/results/ ; large files: /mnt/local/aiskylimit_new_nothing/selot/work/.
# Test runs elsewhere (e.g. Colab, colab/smoke_new_nothing.ipynb) override the defaults with SELOT_* variables
# (SELOT_BASE, SELOT_VENV=none, SELOT_MODELS_DIR, SELOT_WORK, SELOT_RESULTS_ROOT, SELOT_RAW_SRC, SELOT_GPU_IDS,
# SELOT_LAYERS, SELOT_MIN_FREE_MB, SELOT_SMOKE=1: few prompts per set, SELOT_SKIP_DATA=1: keep data/). On the server
# none is set.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

PROJECT_NAME="aiskylimit_new_nothing"
BASE="${SELOT_BASE:-/mnt/local/${PROJECT_NAME}/selot}"   # must match download.txt
VENV="${SELOT_VENV:-/mnt/local/uvenvs/selot}"             # built from ../selot.txt by the env job
GPU_IDS="${SELOT_GPU_IDS:-2,3}"                           # the GPUs SElOT may use on this server
MODEL="llama3.1-8b"
LAYERS="${SELOT_LAYERS:-}"         # SElOT sweep layers; empty = model.layer_range of configs/base.yaml (30-70%: 10-22)
SWEEP_ALPHA="1.0"
MIN_FREE_MB="${SELOT_MIN_FREE_MB:-60000}"                 # per GPU, checked before every run
SMOKE="${SELOT_SMOKE:-0}"                                 # 1 = test run: few prompts per set, *_smoke runs

if [ "${VENV}" != "none" ]; then
  if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: ${VENV} does not exist. Run the env job first: header '#i selot.txt' in commands.sh" >&2
    exit 1
  fi
  source "${VENV}/bin/activate"
fi
export PYTHONPATH=.

export MODELS_DIR="${SELOT_MODELS_DIR:-${BASE}/models}" WORK="${SELOT_WORK:-${BASE}/work}"
export RESULTS_ROOT="${SELOT_RESULTS_ROOT:-${ROOT}/results}" RAW_SRC="${SELOT_RAW_SRC:-${BASE}/raw_src}"
export GPU_IDS MODEL LAYERS SWEEP_ALPHA MIN_FREE_MB SMOKE
export HF_HOME="${BASE}/.cache/huggingface" VLLM_CACHE_ROOT="${BASE}/.cache/vllm" TMPDIR="${BASE}/tmp"
mkdir -p "${RESULTS_ROOT}/logs" "${WORK}" "${TMPDIR}" "${VLLM_CACHE_ROOT}"
# offline, and nothing phones home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 WANDB_MODE=disabled VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# the data build reads these (offline embedding model, AlphaSteer's prompt files)
export SELOT_EMB_MODEL="${MODELS_DIR}/all-MiniLM-L6-v2" SELOT_ALPHASTEER_DIR="${RAW_SRC}/AlphaLab-USTC__AlphaSteer"

exec > >(tee -a "${RESULTS_ROOT}/logs/job_$(date +%Y%m%d_%H%M%S).log") 2>&1
T0=$(date +%s)
on_exit() {
  local code=$?
  echo
  if [ "${code}" != "0" ]; then
    echo "=== FAILED/STOPPED (exit ${code}) after $(( $(date +%s) - T0 ))s; logs: ${RESULTS_ROOT}/logs and ${RESULTS_ROOT}/results_*/logs ==="
  fi
  for f in "${RESULTS_ROOT}"/sweep_selot_wf1*/sweep.md "${RESULTS_ROOT}"/compare_wf1*.md; do
    [ -f "${f}" ] && { echo "--- ${f}"; cat "${f}"; }
  done
  echo "results: ${RESULTS_ROOT}"
}
trap on_exit EXIT
trap 'exit 143' TERM INT

# inputs: every model folder and the downloaded source files
missing=()
for m in Llama-3.1-8B-Instruct Llama-Guard-3-8B HarmBench-Llama-2-13b-cls gemma-2b strongreject-15k-v1 \
         Skywork-Reward-V2-Llama-3.1-8B Qwen2.5-14B-Instruct all-MiniLM-L6-v2; do
  [ -e "${MODELS_DIR}/${m}/config.json" ] || [ -e "${MODELS_DIR}/${m}/adapter_config.json" ] || missing+=("${MODELS_DIR}/${m}")
done
[ -d "${RAW_SRC}" ] || missing+=("${RAW_SRC} (source data files)")
if [ ${#missing[@]} -gt 0 ]; then
  printf 'ERROR: missing (run the download job: header = the content of SElOT/download.txt):\n'; printf '  %s\n' "${missing[@]}"
  exit 1
fi
for g in ${GPU_IDS//,/ }; do
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${g}" | tr -d ' ')
  echo "GPU ${g}: ${FREE_MB} MiB free"
  if [ "${FREE_MB}" -lt "${MIN_FREE_MB}" ]; then
    echo "ERROR: GPU ${g} has only ${FREE_MB} MiB free (need ${MIN_FREE_MB})" >&2
    nvidia-smi
    exit 1
  fi
done

if [ "${SELOT_SKIP_DATA:-0}" = "1" ]; then   # test runs only (resuming a smoke run): keep the data/ already there
  [ -d data/fit ] && [ -d data/eval ] || { echo "ERROR: SELOT_SKIP_DATA=1 but data/fit or data/eval is missing" >&2; exit 1; }
  echo; echo "=== data: SELOT_SKIP_DATA=1, using the existing data/fit and data/eval ==="
else
  echo; echo "=== data: rebuilt from ${RAW_SRC} (kept between jobs: only the MiniLM embedding cache data/emb) ==="
  rm -rf data/raw data/fit data/eval data/reports
  python scripts/fetch_data.py --src "${RAW_SRC}" --force
  python scripts/prepare_eval.py
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" python scripts/build_fit_sets.py --device cuda   # MiniLM on our first GPU
  python scripts/data_manifest.py --check
fi

echo; echo "=== pipeline ==="
python scripts/pipeline.py
echo "ALL DONE - $(( $(date +%s) - T0 ))s"
