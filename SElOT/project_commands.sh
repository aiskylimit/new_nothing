#!/bin/bash
# SElOT job on the new_nothing server (offline: no outbound calls; models and source files come from the download job).
# commands.sh runs exactly "cd ./SElOT" + "bash ./project_commands.sh", so every default below must be right as is;
# to change a setting, edit it here. Resubmit the same job after a timeout: every step resumes (see RUN.md).
#
# Every job: rebuild the data from the downloaded source files (fetch_data --src, prepare_eval, build_fit_sets; checked
# against data_manifest.json), then scripts/pipeline.py:
#   label (fit-data workflow WF; now 1 = the first Colab run) -> activations at layers 8-22 (sweep 10-22
#   + AlphaSteer's 8-19) -> layer sweep of METHOD on the val split (layers 10-22 = 30-70% of depth, alpha 1)
#   -> best layer by mean val DSR (results/sweep_<METHOD>_wf<WF>/sweep.md; activation plots of the best layer's map in
#   results/results_sweep_<METHOD>_wf<WF>_L<best>/plots when matplotlib is installed)
#   [UNTIL=all only: -> METHOD full eval -> AlphaSteer full eval -> results/compare_wf<WF>[_<METHOD>].md (final
#   tables: safety, utility) + results/tune_log_wf<WF>[_<METHOD>].md (sweep + every run's details)]
# METHOD: selot_gated = SElOT-gated (configs/methods/selot_gated.yaml: a Gaussian-mixture gate decides whether to steer,
#   one Gaussian OT per jailbreak mode where; components by BIC), selot = SElOT (ridge field).
# Current job (defaults below): SElOT-gated, workflow 1, full pipeline (UNTIL=all): layer sweep, SElOT-gated full
# eval at the best layer, AlphaSteer full eval (baseline), final tables.
# No W&B. Small outputs: SElOT/results/ ; large files: /mnt/local/aiskylimit_new_nothing/selot/work/.
# Test runs elsewhere (e.g. Colab, colab/smoke_new_nothing.ipynb) override the defaults with SELOT_* variables
# (SELOT_BASE, SELOT_VENV=none, SELOT_MODELS_DIR, SELOT_WORK, SELOT_RESULTS_ROOT, SELOT_RAW_SRC, SELOT_GPU_IDS,
# SELOT_LAYERS, SELOT_MIN_FREE_MB, SELOT_SMOKE=1: few prompts per set, SELOT_SKIP_DATA=1: keep data/, SELOT_WF,
# SELOT_UNTIL, SELOT_METHOD). On the server
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
SELOT_WF="${SELOT_WF:-1}"                                 # fit-data workflow: 1 = first Colab run | 2 = split by behaviour group
SELOT_UNTIL="${SELOT_UNTIL:-all}"                         # all = + full evals (METHOD, AlphaSteer) + tables | select = stop after the best layer
SELOT_METHOD="${SELOT_METHOD:-selot_gated}"               # steering method: selot_gated (gate + per-mode OT) | selot
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
export GPU_IDS MODEL LAYERS SWEEP_ALPHA MIN_FREE_MB SMOKE SELOT_WF SELOT_UNTIL SELOT_METHOD
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
  # only this job's files (its workflow, _smoke or not): the final tables when they exist, else the layer sweep
  local sfx="${SELOT_WF}$([ "${SMOKE}" = "1" ] && echo _smoke || true)"
  local rsfx="${sfx}$([ "${SELOT_METHOD}" = "selot" ] || echo "_${SELOT_METHOD}")"
  if [ "${code}" = "0" ] && [ "${SELOT_UNTIL}" = "all" ] && [ -f "${RESULTS_ROOT}/compare_wf${rsfx}.md" ]; then
    echo "--- ${RESULTS_ROOT}/compare_wf${rsfx}.md"; cat "${RESULTS_ROOT}/compare_wf${rsfx}.md"
  elif [ -f "${RESULTS_ROOT}/sweep_${SELOT_METHOD}_wf${sfx}/sweep.md" ]; then
    echo "--- layer sweep: ${RESULTS_ROOT}/sweep_${SELOT_METHOD}_wf${sfx}/sweep.md"
    cat "${RESULTS_ROOT}/sweep_${SELOT_METHOD}_wf${sfx}/sweep.md"
  fi
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
# chat templates (tokenizer_config.json or chat_template.jinja): checked here, not only at the judge stage hours later
for m in Llama-3.1-8B-Instruct Llama-Guard-3-8B Qwen2.5-14B-Instruct Skywork-Reward-V2-Llama-3.1-8B; do
  d="${MODELS_DIR}/${m}"
  [ -e "${d}/config.json" ] || continue
  [ -e "${d}/chat_template.jinja" ] || [ -e "${d}/chat_template.json" ] || grep -q '"chat_template"' "${d}/tokenizer_config.json" 2>/dev/null \
    || missing+=("${d}: no chat template (tokenizer_config.json / chat_template.jinja): download the whole model repo")
done
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
