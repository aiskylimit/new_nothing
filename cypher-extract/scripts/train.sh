#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train.sh <config.yaml> [key=value ...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
shift
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
fi

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPU_LIST <<< "${RUN_GPUS}"
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_LIST[*]}")"
  NPROC_PER_NODE="${#GPU_LIST[@]}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export FORCE_TORCHRUN=1
export ACCELERATE_MIXED_PRECISION=bf16
cd "${PROJECT_ROOT}"

AUTO_DATA_OVERRIDES=()
if [[ "${AUTO_PREPARE_DATA:-1}" == "1" ]]; then
  MANAGED_DATASET_DIR="$(python -m distillation.auto_prepare "${CONFIG_PATH}" "$@")"
  if [[ -n "${MANAGED_DATASET_DIR}" ]]; then
    AUTO_DATA_OVERRIDES+=("dataset_dir=${MANAGED_DATASET_DIR}")
  fi
fi

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES:-1}" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${RUN_MASTER_ADDR:-${MASTER_ADDR:-localhost}}" \
  --master_port "${RUN_MASTER_PORT:-${MASTER_PORT:-29500}}" \
  -m distillation.cli "${CONFIG_PATH}" "$@" "${AUTO_DATA_OVERRIDES[@]}"
