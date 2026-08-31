#!/usr/bin/env bash
# Phase 3 (Pru-CoT baseline): step-importance global optimization for the Qwen3-4B track.
set -euo pipefail

GPUS=(6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothing}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-4B"
DATA_PATH="data/qwen3-4b/train-segmented.jsonl"
OUTPUT_DIR="data/qwen3-4b/prucot"
WEIGHTS_PATH="data/qwen3-4b/prucot-weights.parquet"
EPOCHS=3
LR=10
FILLER_TOKEN="."
# Empty = keep all; set to drop longer records (paper-style). Must match prune_qwen3-4b.sh's value.
MAX_LENGTH="${PRUCOT_MAX_LENGTH:-}"

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --weights-path ${WEIGHTS_PATH}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --lr ${LR}"
OPTS+=" --filler-token ${FILLER_TOKEN}"
[[ -n "${MAX_LENGTH}" ]] && OPTS+=" --max-length ${MAX_LENGTH}"

NUM_SHARDS=${#GPUS[@]}
# One compute shard per GPU (no in-process multi-GPU path); final --num-shards 1 pass merges.
echo ">>> launching ${NUM_SHARDS} shards (one per GPU: ${GPUS[*]}) of ${BASE_PATH}/src/prucot_weight.py"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python -u "${BASE_PATH}/src/prucot_weight.py" ${OPTS} \
    --num-shards "${NUM_SHARDS}" --shard-index "${i}" \
    > "logs/qwen3-4b-prucot-weight-shard${i}.log" 2>&1 &
  pids+=($!)
done
shard_fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "shard ${i} failed -- see logs/qwen3-4b-prucot-weight-shard${i}.log" >&2; shard_fail=1; }
done
[[ ${shard_fail} -eq 0 ]] || exit 1

# Merge pass: every npz exists, so this only rebuilds the parquet (still loads the model).
CMD="python -u ${BASE_PATH}/src/prucot_weight.py ${OPTS}"
echo "${CMD}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" ${CMD} 2>&1 | tee logs/qwen3-4b-prucot-weight.log
