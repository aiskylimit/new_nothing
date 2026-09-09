#!/usr/bin/env bash
set -euo pipefail

# All benchmark assets are downloaded by download.txt before this script runs.
# Usage:
#   bash scripts/eval/eval.sh check
#   bash scripts/eval/eval.sh run

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
# Prevent the project-level evaluate.py from shadowing Hugging Face Evaluate.
unset PYTHONPATH

ACTION="${1:-run}"
if [[ "$ACTION" != "check" && "$ACTION" != "run" ]]; then
    printf 'Usage: %s [check|run]\n' "$0" >&2
    exit 2
fi

if [[ -n "${EVAL_PYTHON:-}" ]]; then
    PYTHON_BIN="$EVAL_PYTHON"
elif [[ -n "${EVAL_VENV_PATH:-}" ]]; then
    PYTHON_BIN="$EVAL_VENV_PATH/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    PYTHON_BIN="/mnt/local/uvenvs/reasoning-velocity-distill/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Evaluation Python not found: %s\n' "$PYTHON_BIN" >&2
    printf 'Set EVAL_PYTHON or EVAL_VENV_PATH to the lm-evaluation-harness environment.\n' >&2
    exit 1
fi

export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$BASE_PATH/data/eval}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$BASE_PATH/.cache/eval}"
export HF_HOME="${EVAL_HF_HOME:-$EVAL_CACHE_DIR/huggingface}"
export HF_DATASETS_CACHE="${EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${EVAL_MODULES_CACHE:-$EVAL_CACHE_DIR/modules}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

EVAL_TASKS="${EVAL_TASKS:-gsm8k_cot,gsm_plus,minerva_math,mbpp_instruct,sciq,mmlu_stem,mmlu_pro_math,bbh_cot_fewshot}"
EVAL_TASKS="${EVAL_TASKS// /,}"

check_harness() {
    "$PYTHON_BIN" -c '
from importlib.metadata import version

expected = {"lm_eval": "0.4.13", "evaluate": "0.4.6"}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"Expected {package}=={wanted}, found {actual}")
' || {
        printf 'Install lm_eval[hf,math]==0.4.13 and evaluate==0.4.6 in the evaluation environment.\n' >&2
        exit 1
    }
}

check_local_data() {
    check_harness
    mkdir -p -- "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"
    "$PYTHON_BIN" scripts/eval/local_lm_eval.py check --tasks "$EVAL_TASKS"
}

resolve_lora_path() {
    if [[ -n "${LORA_PATH:-}" ]]; then
        return
    fi

    local args_file="$SAVE_PATH/args.json"
    if [[ ! -f "$args_file" ]]; then
        printf 'Training metadata not found: %s\n' "$args_file" >&2
        printf 'Finish training or set LORA_PATH explicitly.\n' >&2
        exit 1
    fi

    local final_step
    final_step="$("$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["total_iters"])' "$args_file")"
    LORA_PATH="$SAVE_PATH/$final_step"
}

print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

run_benchmarks() {
    check_local_data

    MODEL_PATH="${MODEL_PATH:-${CKPT:-$BASE_PATH/models/Qwen2.5_1.5B-Instruct}}"
    SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-rvd}"
    if [[ ! -d "$MODEL_PATH" ]]; then
        printf 'Base model directory not found: %s\n' "$MODEL_PATH" >&2
        exit 1
    fi

    local model_args="pretrained=$MODEL_PATH,dtype=${EVAL_DTYPE:-bfloat16},max_length=${EVAL_MAX_LENGTH:-8192}"
    local label="base"
    if [[ "${EVAL_BASE_ONLY:-0}" != "1" ]]; then
        resolve_lora_path
        if [[ ! -f "$LORA_PATH/adapter_config.json" ]]; then
            printf 'LoRA checkpoint is invalid or incomplete: %s\n' "$LORA_PATH" >&2
            exit 1
        fi
        model_args+=",peft=$LORA_PATH"
        label="step-$(basename -- "$LORA_PATH")"
    fi

    # Evaluation is single-GPU. Use EVAL_GPU when set, otherwise the first GPU
    # selected for the pipeline or supplied through CUDA_VISIBLE_DEVICES.
    local available_gpu_ids="${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0}}"
    export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-${available_gpu_ids%%,*}}"
    local output_dir="${EVAL_OUTPUT_DIR:-$SAVE_PATH/evaluation/$label-benchmarks}"
    local log_dir="${EVAL_LOG_DIR:-$SAVE_PATH/evaluation/logs}"
    local log_file="$log_dir/$label.log"
    mkdir -p -- "$output_dir" "$log_dir"

    local common_args=(
        "$PYTHON_BIN" scripts/eval/local_lm_eval.py run
        --model hf
        --model_args "$model_args"
        --device cuda:0
        --batch_size "${EVAL_BATCH_SIZE:-1}"
        --apply_chat_template
        --fewshot_as_multiturn
        --gen_kwargs "max_gen_toks=${EVAL_MAX_NEW_TOKENS:-2048},do_sample=False"
        --log_samples
        --confirm_run_unsafe_code
    )
    if [[ -n "${EVAL_LIMIT:-}" ]]; then
        common_args+=(--limit "$EVAL_LIMIT")
    fi

    local requested_tasks=()
    local standard_tasks=()
    local mmlu_tasks=()
    IFS=',' read -r -a requested_tasks <<< "$EVAL_TASKS"
    local task
    for task in "${requested_tasks[@]}"; do
        [[ -n "$task" ]] || continue
        if [[ "$task" == "mmlu_stem" ]]; then
            mmlu_tasks+=("$task")
        else
            standard_tasks+=("$task")
        fi
    done

    run_task_group() {
        local group_name="$1"
        local tasks_csv="$2"
        shift 2
        local command=(
            "${common_args[@]}"
            --tasks "$tasks_csv"
            --output_path "$output_dir/$group_name"
            "$@"
        )
        print_command "${command[@]}"
        if [[ "${DRY_RUN:-0}" != "1" ]]; then
            "${command[@]}"
        fi
    }

    {
        printf '[%s] Evaluating %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$label"
        if (( ${#standard_tasks[@]} )); then
            local standard_csv
            standard_csv="$(IFS=,; printf '%s' "${standard_tasks[*]}")"
            run_task_group standard "$standard_csv"
        fi
        if (( ${#mmlu_tasks[@]} )); then
            local mmlu_csv
            mmlu_csv="$(IFS=,; printf '%s' "${mmlu_tasks[*]}")"
            run_task_group mmlu_stem "$mmlu_csv" --num_fewshot "${MMLU_NUM_FEWSHOT:-5}"
        fi
    } 2>&1 | tee "$log_file"
}

case "$ACTION" in
    check) check_local_data ;;
    run) run_benchmarks ;;
esac
