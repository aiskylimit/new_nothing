#!/usr/bin/env bash
# Pipeline driver. Two phases: TRAIN everything, then EVAL everything.
#   TRAIN: data (shared) -> spectral -> vanilla -> prucot   (per method, per model)
#   EVAL : only after all training is done; only checkpoints that trained OK.
# Isolation: a failed block is recorded and the driver keeps going. A `data` failure skips that
# model's later blocks; a training failure skips only that (method, model)'s eval.
set -uo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=(qwen3-1.7b qwen3-4b)
METHODS=(spectral vanilla prucot)
FAILED_TRACKS=()
declare -A BROKEN      # model -> data prep failed
declare -A TRAINED     # "<method>-<model>" -> checkpoint trained OK (eligible for eval)

run_step() {
  local label="$1"
  shift
  ( "$@" )
  local status=$?
  if [[ ${status} -eq 0 ]]; then
    echo ">>> ${label}: OK"
    return 0
  fi
  echo ">>> ${label}: FAILED (exit ${status}) -- continuing" >&2
  FAILED_TRACKS+=("${label}")
  return 1
}

# ---- training blocks (NO eval; each phase in order) ----
prep() {
  set -e
  "${BASE_PATH}/scripts/qwen3/data/data_$1.sh"
}

spectral_train() {
  set -e
  "${BASE_PATH}/scripts/qwen3/data/capture_$1.sh"
  "${BASE_PATH}/scripts/qwen3/data/masks_$1.sh"
  "${BASE_PATH}/scripts/qwen3/spectral/spectral_$1.sh"
}

vanilla_train() {
  set -e
  "${BASE_PATH}/scripts/qwen3/sft/sft_$1.sh"
}

prucot_train() {
  set -e
  "${BASE_PATH}/scripts/qwen3/prucot/weight_$1.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/prune_$1.sh"
  "${BASE_PATH}/scripts/qwen3/prucot/prucot_$1.sh"
}

# eval one (method, model). Default (no args) evaluates the spectral checkpoint.
eval_ckpt() {
  local method="$1" m="$2"
  if [[ "${method}" == spectral ]]; then
    "${BASE_PATH}/scripts/qwen3/eval/eval_$m.sh"
  else
    "${BASE_PATH}/scripts/qwen3/eval/eval_$m.sh" "${BASE_PATH}/checkpoints/${method}-$m" "${method}-$m"
  fi
}

# ================= TRAIN PHASE =================
echo "===== TRAIN PHASE ====="

for m in "${MODELS[@]}"; do
  run_step "data-${m}" prep "${m}" || BROKEN[$m]=1
done

for m in "${MODELS[@]}"; do
  [[ -n "${BROKEN[$m]:-}" ]] && { echo ">>> spectral-train-${m}: SKIPPED (data failed)" >&2; continue; }
  run_step "spectral-train-${m}" spectral_train "${m}" && TRAINED[spectral-$m]=1
done

for m in "${MODELS[@]}"; do
  [[ -n "${BROKEN[$m]:-}" ]] && { echo ">>> vanilla-train-${m}: SKIPPED (data failed)" >&2; continue; }
  run_step "vanilla-train-${m}" vanilla_train "${m}" && TRAINED[vanilla-$m]=1
done

# Pru-CoT baseline disabled for now -- uncomment this loop to re-enable (eval auto-includes it).
# for m in "${MODELS[@]}"; do
#   [[ -n "${BROKEN[$m]:-}" ]] && { echo ">>> prucot-train-${m}: SKIPPED (data failed)" >&2; continue; }
#   run_step "prucot-train-${m}" prucot_train "${m}" && TRAINED[prucot-$m]=1
# done

# ================= EVAL PHASE =================
echo "===== EVAL PHASE (all training complete) ====="

for m in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    if [[ -z "${TRAINED[${method}-$m]:-}" ]]; then
      echo ">>> eval-${method}-${m}: SKIPPED (${method}-${m} did not train)" >&2
      continue
    fi
    run_step "eval-${method}-${m}" eval_ckpt "${method}" "${m}"
  done
done

python "${BASE_PATH}/src/compare_results.py" || echo ">>> compare_results.py failed" >&2

if [[ ${#FAILED_TRACKS[@]} -gt 0 ]]; then
  echo ">>> failed blocks: ${FAILED_TRACKS[*]}" >&2
  exit 1
fi
