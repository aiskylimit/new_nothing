#!/usr/bin/env bash
# Phase 3: gradient capture (--verify) for the Qwen3-4B-Instruct-2507 track.
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
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-4B-Instruct-2507"
DATA_PATH="data/qwen3-4b-instruct/train-s1k-segmented.jsonl"
OUTPUT_DIR="data/qwen3-4b-instruct/spectral"
STRENGTHS_PATH="data/qwen3-4b-instruct/spectral-strengths.parquet"
ENERGY_CUTOFF=0.95
CHUNK_SIZE=1024

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --strengths-path ${STRENGTHS_PATH}"
OPTS+=" --energy-cutoff ${ENERGY_CUTOFF}"
OPTS+=" --chunk-size ${CHUNK_SIZE}"

NUM_SHARDS=${#GPUS[@]}
# gradient_capture.py has no in-process multi-GPU path: one sample runs on one GPU. To actually use every GPU
# in GPUS we launch one shard per GPU -- --num-shards/--shard-index split the corpus into disjoint
# subsets, each shard pins itself to a single GPU and writes its own per-sample npz (--verify runs on shard 0 only, per gradient_capture.py's shard-index guard). A
# final --num-shards 1 pass then hits the resume path for every npz and rebuilds the parquet.
echo ">>> launching ${NUM_SHARDS} shards (one per GPU: ${GPUS[*]}) of ${BASE_PATH}/src/gradient_capture.py"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python "${BASE_PATH}/src/gradient_capture.py" ${OPTS} --verify \
    --num-shards "${NUM_SHARDS}" --shard-index "${i}" \
    > "logs/qwen3-4b-instruct-capture-shard${i}.log" 2>&1 &
  pids+=($!)
done
shard_fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "shard ${i} failed -- see logs/qwen3-4b-instruct-capture-shard${i}.log" >&2; shard_fail=1; }
done
[[ ${shard_fail} -eq 0 ]] || exit 1

# Merge pass (--num-shards 1, default): all per-sample npz now exist, so every record resumes
# from disk and this just rebuilds the parquet. Still loads the model, so give it one GPU.
CMD="python ${BASE_PATH}/src/gradient_capture.py ${OPTS}"
echo "${CMD}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" ${CMD} 2>&1 | tee logs/qwen3-4b-instruct-capture.log

echo ">>> STOP AND READ: check 'k*/T mean ratio' and 'strength spread' above (docs/server-runbook.md §5)."
