#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/experiment_settings.env"

MODEL_FILTER="${1:-}"
METHOD_FILTER="${2:-}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-${PROJECT_ROOT}/data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

DATASETS=(
    # aime24
    aime25
    aime26
    hmmt25
)
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
OPSD_EVAL_STEPS="${OPSD_EVAL_STEPS:-25 50 75 100}"
GRPO_EVAL_STEPS="${GRPO_EVAL_STEPS:-400 425 450 500}"

evaluate_checkpoint() {
    local model_size="$1"
    local method="$2"
    local step="$3"
    local model_path="$4"
    local checkpoint_path="$5"

    for dataset in "${DATASETS[@]}"; do
        local output_dir="${RESULTS_ROOT}/raw/${model_tag}/${method}/${step}"
        local output_file="${output_dir}/${dataset}.json"
        mkdir -p "${output_dir}"

        if [[ -f "${output_file}" && "${OVERWRITE_EVAL}" != "1" ]]; then
            echo "Skipping existing result: ${output_file}"
            continue
        fi

        local checkpoint_args=()
        if [[ -n "${checkpoint_path}" ]]; then
            checkpoint_args=(--checkpoint_dir "${checkpoint_path}")
        fi

        python "${PROJECT_ROOT}/eval/evaluate_math.py" \
            --base_model "${model_path}" \
            --dataset_dir "${PREPARED_DATA_ROOT}/eval" \
            --dataset "${dataset}" \
            --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
            --temperature "${EVAL_TEMPERATURE}" \
            --top_p "${EVAL_TOP_P}" \
            --top_k "${EVAL_TOP_K}" \
            --min_p 0 \
            --presence_penalty 0 \
            --val_n "${EVAL_VAL_N}" \
            --tensor_parallel_size "${EVAL_TENSOR_PARALLEL_SIZE}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
            --output_file "${output_file}" \
            "${checkpoint_args[@]}"
    done
}

for model_size in 4b 8b olmo7b; do
    if [[ -n "${MODEL_FILTER}" && "${model_size}" != "${MODEL_FILTER}" ]]; then
        continue
    fi

    case "${model_size}" in
        4b|8b)
            model_path="${MODEL_ROOT}/Qwen3-${model_size^^}"
            model_tag="qwen3-${model_size}"
            run_tag="qwen3_${model_size}"
            ;;
        olmo7b)
            model_path="${MODEL_ROOT}/Olmo-3-7B-Think"
            model_tag="olmo3-7b-think"
            run_tag="olmo3_7b_think"
            ;;
    esac
    if [[ ! -d "${model_path}" ]]; then
        echo "Missing local model: ${model_path}" >&2
        exit 1
    fi

    if [[ -z "${METHOD_FILTER}" || "${METHOD_FILTER}" == "base" ]]; then
        OVERWRITE_EVAL=0 evaluate_checkpoint "${model_size}" base base "${model_path}" ""
    fi

    for method in sft opsd grpo; do
        if [[ -n "${METHOD_FILTER}" && "${method}" != "${METHOD_FILTER}" ]]; then
            continue
        fi

        run_dir="${OUTPUT_ROOT}/${method}/${method}_${run_tag}_paper"
        case "${method}" in
            sft) steps="${SFT_EVAL_STEPS}" ;;
            opsd) steps="${OPSD_EVAL_STEPS}" ;;
            grpo) steps="${GRPO_EVAL_STEPS}" ;;
        esac

        for step in ${steps}; do
            checkpoint_path="${run_dir}/checkpoint-${step}"
            if [[ ! -d "${checkpoint_path}" ]]; then
                echo "Skipping missing checkpoint: ${checkpoint_path}"
                continue
            fi
            evaluate_checkpoint "${model_size}" "${method}" "${step}" "${model_path}" "${checkpoint_path}"
        done
    done
done

python "${PROJECT_ROOT}/eval/summarize_results.py" \
    --results_root "${RESULTS_ROOT}/raw" \
    --output_dir "${RESULTS_ROOT}"
