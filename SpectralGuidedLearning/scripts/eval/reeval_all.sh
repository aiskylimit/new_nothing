#!/usr/bin/env bash
# Re-eval EVERY trained arm of the qwen25-7b / qwen3-8b tracks under the 32k generation cap.
#   bash scripts/eval/reeval_all.sh                 # all arms, both tracks
#   TRACKS="qwen3-8b" bash scripts/eval/reeval_all.sh
#   EVAL_BASE=1 bash scripts/eval/reeval_all.sh     # also eval the untrained base models as a reference
#
# Arms are discovered from checkpoints/<arm>-<track>/ (adapter-only or merged); each is evaluated
# with the track's eval script into RESULTS_DIR (default results-32k/, so the original 3.5k-cap
# results/ tree is kept for comparison). Arms whose summary.json is already in RESULTS_DIR are
# skipped, so a crashed driver can simply be re-run; delete <RESULTS_DIR>/<tag>/ to force one.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
export PYTHONPATH="${BASE_PATH}/src${PYTHONPATH:+:${PYTHONPATH}}"
export GPUS="${GPUS:-0}"
export RESULTS_DIR="${RESULTS_DIR:-${BASE_PATH}/results-32k}"
TRACKS="${TRACKS:-qwen25-7b qwen3-8b}"
EVAL_BASE="${EVAL_BASE:-0}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
mkdir -p "${RESULTS_DIR}" logs

# The eval scripts switch to the vLLM env themselves; never inherit the unsloth train env.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

run_eval() {  # run_eval <track> <model path> <tag>
  local track="$1" model="$2" tag="$3"
  if [[ -f "${RESULTS_DIR}/${tag}/summary.json" ]]; then
    echo "[skip] ${tag}: ${RESULTS_DIR}/${tag}/summary.json exists"
    return
  fi
  echo "[eval] ${tag} <- ${model}"
  bash "scripts/eval/eval_${track}.sh" "${model}" "${tag}"
}

for track in ${TRACKS}; do
  case "${track}" in
    qwen25-7b) base_dir="Qwen2.5-7B-Instruct" ;;
    qwen3-8b) base_dir="Qwen3-8B" ;;
    *) echo "unknown track: ${track}" >&2; exit 2 ;;
  esac
  if [[ "${EVAL_BASE}" == 1 ]]; then
    run_eval "${track}" "${LOCAL_MODELS_ROOT}/${base_dir}" "base-${track}"
  fi
  found=0
  for ckpt in checkpoints/*-"${track}"; do
    # a finished run has the adapter (--no-lora-merge) or a merged model at its root;
    # intermediate checkpoint-N/ dirs are not evaluated
    [[ -f "${ckpt}/adapter_config.json" || -f "${ckpt}/config.json" ]] || continue
    found=1
    run_eval "${track}" "${ckpt}" "$(basename "${ckpt}")"
  done
  [[ "${found}" == 1 ]] || echo "[warn] no finished checkpoints matched checkpoints/*-${track}" >&2
done

# regenerated from every <RESULTS_DIR>/<tag>/summary.json -> comparison-table.md + eval-summary.json
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" -m sgl.eval.compare --results-dir "${RESULTS_DIR}"
cat "${RESULTS_DIR}/comparison-table.md"
