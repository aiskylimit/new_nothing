#!/usr/bin/env bash
set -euo pipefail

inference_root="${INFERENCE_ROOT:-results/inference/qwen3}"
evaluation_root="${EVALUATION_ROOT:-results/evaluation/qwen3}"
seed_csv="${SEEDS:-10,42,50,100,1234}"
method_csv="${METHODS:-}"
dataset_csv="${DATASETS:-cypherbench,mind_the_query,neo4j_text2cypher}"
metric_csv="${METRICS:-execution_accuracy,psjs,executable}"
python_bin="${PYTHON_BIN:-python}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
if [[ "$inference_root" != /* ]]; then
    inference_root="$repository_root/$inference_root"
fi
if [[ "$evaluation_root" != /* ]]; then
    evaluation_root="$repository_root/$evaluation_root"
fi
export PYTHONPATH="$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a seeds <<< "$seed_csv"
IFS=',' read -r -a datasets <<< "$dataset_csv"
IFS=',' read -r -a metrics <<< "$metric_csv"

dataset_graphs() {
    case "$1" in
        cypherbench)
            echo "company fictional_character flight_accident geography movie nba politics"
            ;;
        mind_the_query)
            echo "bloom50 healthcare wwc"
            ;;
        neo4j_text2cypher)
            echo "bluesky buzzoverflow companies fincen gameofthrones grandstack movies neoflix network northwind offshoreleaks recommendations stackoverflow2 twitch twitter"
            ;;
        *)
            echo "Unknown dataset '$1'." >&2
            return 1
            ;;
    esac
}

dataset_connector() {
    case "$1" in
        cypherbench) echo "cypherbench-db" ;;
        mind_the_query) echo "mind-the-query-db" ;;
        neo4j_text2cypher) echo "neo4j_text2cypher_db" ;;
        *) return 1 ;;
    esac
}

for seed in "${seeds[@]}"; do
    seed_input="$inference_root/seed$seed"
    if [[ ! -d "$seed_input" ]]; then
        echo "Inference seed directory does not exist: $seed_input" >&2
        exit 1
    fi

    if [[ -n "$method_csv" ]]; then
        IFS=',' read -r -a seed_methods <<< "$method_csv"
    else
        mapfile -t seed_methods < <(find "$seed_input" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    fi
    if (( ${#seed_methods[@]} == 0 )); then
        echo "No method directories found under $seed_input" >&2
        exit 1
    fi

    echo "[seed$seed] evaluating methods: ${seed_methods[*]}"
    for method in "${seed_methods[@]}"; do
        for dataset in "${datasets[@]}"; do
            input_path="$seed_input/$method/$dataset/generator_predictions.jsonl"
            if [[ ! -f "$input_path" ]]; then
                echo "Missing inference output: $input_path" >&2
                exit 1
            fi

            dataset_output="$evaluation_root/seed$seed/$method/$dataset"
            read -r -a graphs <<< "$(dataset_graphs "$dataset")"
            connector="$(dataset_connector "$dataset")"
            for graph in "${graphs[@]}"; do
                output_path="$dataset_output/$graph/cypher_scores.jsonl"
                echo "[seed$seed/$method/$dataset/$graph] evaluating"
                "$python_bin" -m cypher_evaluation.cli \
                    --input "$input_path" \
                    --output "$output_path" \
                    --name "$connector" \
                    --graph "$graph" \
                    --metrics "${metrics[@]}"
            done

            "$python_bin" -m cypher_evaluation.merge \
                --input-dir "$dataset_output" \
                --metrics "${metrics[@]}"
        done
    done
    echo "[seed$seed] complete"
done
