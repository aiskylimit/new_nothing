#!/usr/bin/env bash
# End-to-end local/offline SR-EOTD training and evaluation.
#
# Usage:
#   bash project_commands.sh              # environment check + train + eval
#   bash project_commands.sh env
#   bash project_commands.sh check
#   bash project_commands.sh train
#   bash project_commands.sh eval         # evaluate an existing run
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1 GPU_NUM=2 bash project_commands.sh all
#   RUN_DIR='results/gpt2/train/<run>' bash project_commands.sh eval

# Chế độ bash nghiêm ngặt
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Chọn action và activate virtual environment
ACTION="${1:-all}"
VENV_PATH="${SREOTD_VENV:-/mnt/local/uvenvs/sreotd}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    printf 'Python environment not found: %s\n' "$VENV_PATH" >&2
    printf 'Set SREOTD_VENV to the correct environment directory.\n' >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Setup python, cache, logging, ...
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# All model and dataset reads must remain local after download.txt is processed.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Cấu hình train và eval
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_NUM="${GPU_NUM:-1}"
MASTER_PORT="${MASTER_PORT:-2012}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-2113}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
EVAL_SEEDS="${EVAL_SEEDS:-10 20 30 40 50}"

# Thu stdout/stderr của toàn bộ phiên chạy vào một file riêng.
# shellcheck disable=SC1091
source "$ROOT/run_logging.sh"
start_run_log "$ROOT" "$ACTION"

# Khai báo các local path
STUDENT="$ROOT/results/gpt2/train/init/gpt2-base"
TEACHER="$ROOT/results/gpt2/train/sft/gpt2-xlarge"
DOLLY_PROCESSED="$ROOT/processed_data/dolly/full/gpt2"
OPENWEBTEXT_PROCESSED="$ROOT/processed_data/openwebtext/gpt2/512/10M"
DEFAULT_RUN_DIR="results/gpt2/train/0.1B_1.5B#sreotd/lm_${MICRO_BATCH_SIZE}_${LR}"
RUN_DIR="${RUN_DIR:-$DEFAULT_RUN_DIR}"

usage() {
    printf 'usage: bash project_commands.sh [all|env|check|train|eval]\n' >&2
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        printf 'Missing local asset directory: %s\n' "$1" >&2
        exit 1
    fi
}

cmd_env() {
    python -c 'import accelerate, datasets, deepspeed, peft, torch, transformers; print("environment ok | torch", torch.__version__, "| cuda", torch.cuda.is_available())'
}

cmd_check() {
    require_dir "$STUDENT"
    require_dir "$TEACHER"
    require_dir "$DOLLY_PROCESSED"
    require_dir "$OPENWEBTEXT_PROCESSED"
    require_dir "$ROOT/data/dolly"
    require_dir "$ROOT/data/self-inst"
    require_dir "$ROOT/data/vicuna"
    require_dir "$ROOT/data/sinst"
    require_dir "$ROOT/data/uinst"

    [[ -f "$STUDENT/config.json" ]] || {
        printf 'Missing student config: %s/config.json\n' "$STUDENT" >&2
        exit 1
    }
    [[ -f "$TEACHER/config.json" ]] || {
        printf 'Missing teacher config: %s/config.json\n' "$TEACHER" >&2
        exit 1
    }

    local visible_gpu_count
    visible_gpu_count="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
    if (( GPU_NUM > visible_gpu_count )); then
        printf 'GPU_NUM=%s exceeds CUDA_VISIBLE_DEVICES=%s\n' "$GPU_NUM" "$CUDA_VISIBLE_DEVICES" >&2
        exit 1
    fi

    printf 'local assets ok\n'
}

# Chạy train
cmd_train() {
    printf '\n[train] SR-EOTD GPT-2 XL -> GPT-2 Base\n'
    bash "$ROOT/scripts/gpt2/sreotd/train_0.1B_1.5B.sh" \
        "$ROOT" "$MASTER_PORT" "$GPU_NUM" "$MICRO_BATCH_SIZE" "$LR"
}

# Chạy eval
cmd_eval() {
    require_dir "$ROOT/$RUN_DIR"
    printf '\n[eval] run=%s seeds=%s\n' "$RUN_DIR" "$EVAL_SEEDS"
    EVAL_SEEDS="$EVAL_SEEDS" \
        bash "$ROOT/scripts/gpt2/eval/run_eval.sh" \
        "$RUN_DIR" "$EVAL_MASTER_PORT" \
        --kd-objective sr_eotd \
        --teacher-model-path "$TEACHER" \
        --teacher-model-fp16
}

# Điều hướng action
case "$ACTION" in
    env)
        cmd_env
        ;;
    check)
        cmd_check
        ;;
    train)
        cmd_check
        cmd_train
        ;;
    eval)
        cmd_check
        cmd_eval
        ;;
    all)
        cmd_env
        cmd_check
        cmd_train
        cmd_eval
        ;;
    *)
        printf 'Unknown action: %s\n' "$ACTION" >&2
        usage
        exit 1
        ;;
esac
