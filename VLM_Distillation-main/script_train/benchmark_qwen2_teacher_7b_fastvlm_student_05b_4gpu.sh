#!/usr/bin/env bash
# Fair 4-GPU speed/VRAM benchmark for Qwen2-VL-7B -> FastVLM-0.5B methods.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"

# Environment split used by the existing training setup: all baselines run in
# train-2; only the joint SCVA-SIC method runs in train-3. Each method is
# launched from a subshell below so activating one venv cannot leak into another.
BASELINE_TRAIN_ENV="${BASELINE_TRAIN_ENV:-/mnt/local/uvenvs/vlm-distill-train-2/bin/activate}"
SCVA_SIC_TRAIN_ENV="${SCVA_SIC_TRAIN_ENV:-/mnt/local/uvenvs/vlm-distill-train-3/bin/activate}"

for required_env in "${BASELINE_TRAIN_ENV}" "${SCVA_SIC_TRAIN_ENV}"; do
  if [[ ! -f "${required_env}" ]]; then
    echo "Training environment activation script not found: ${required_env}" >&2
    echo "Override BASELINE_TRAIN_ENV or SCVA_SIC_TRAIN_ENV and run again." >&2
    exit 1
  fi
done

# Physical GPUs are fixed as requested. torchrun sees them as logical 0,1,2,3.
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-4,5,6,7}"
NPROC_PER_NODE=4

MODEL_ROOT="${MODEL_ROOT:-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/models}"
STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT}/KamilaMila/FastVLM-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen/Qwen2-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"

# Fairness controls shared by every method.
PERCENT_DATA="${PERCENT_DATA:-0.001}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
BENCHMARK_WARMUP_STEPS="${BENCHMARK_WARMUP_STEPS:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${PROJECT_DIR}/outputs/benchmark_qwen2_7b_fastvlm_05b_4gpu/${RUN_ID}}"
RESULTS_FILE="${RESULTS_FILE:-${BENCHMARK_ROOT}/benchmark_results.tsv}"
SUMMARY_FILE="${SUMMARY_FILE:-${BENCHMARK_ROOT}/benchmark_summary.tsv}"
# DWA-KD is temporarily excluded from the default suite because its SoftDTW
# CUDA dependency stack is being checked separately. Its case block remains
# available and can be re-enabled explicitly through METHODS.
METHODS="${METHODS:-ce_only emkd sre dskd_v2 mcw_kd scva scva_cgkd scva_sic_low}"

mkdir -p "${BENCHMARK_ROOT}"
# Per-method stdout/stderr is retained separately. The timestamped default root
# prevents a new benchmark from overwriting results from an earlier run.

common_args=(
  --model_name "${STUDENT_MODEL}"
  --teacher_model_name "${TEACHER_MODEL}"
  --data_path "${DATA_PATH}"
  --image_dir "${IMAGE_DIR}"
  --percent_data "${PERCENT_DATA}"
  --lora true
  --lora_r 128
  --lora_alpha 256
  --lora_dropout 0.05
  --per_device_train_batch_size "${PER_DEVICE_BS}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --num_train_epochs 1
  --learning_rate 1e-5
  --weight_decay 0.01
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --bf16 true
  --save_strategy no
  --logging_steps 10
  --dataloader_num_workers "${DATALOADER_WORKERS}"
  --train_sampling_strategy group_by_length
  --max_len 2048
  --image_resolution low
  --resume_from none
  --seed 1337
  --benchmark_enabled true
  --benchmark_warmup_steps "${BENCHMARK_WARMUP_STEPS}"
  --benchmark_output_file "${RESULTS_FILE}"
)

cd "${PROJECT_DIR}"

for method in ${METHODS}; do
  output_dir="${BENCHMARK_ROOT}/${method}"
  log_file="${BENCHMARK_ROOT}/${method}.log"
  method_args=()

  case "${method}" in
    ce_only)
      method_args=(--kd_loss_type ce_only)
      ;;
    emkd)
      method_args=(--kd_loss_type emkd)
      ;;
    sre)
      method_args=(
        --kd_loss_type sre
        --sre_use_projector true
        --teacher_layer_mapping -3 -1
        --student_layer_mapping -3 -1
        --sre_alpha 0.5
        --sre_p 1.0
        --sre_span_loss_weight 1.0
        --sre_geom_loss_weight 3.0
        --sre_logit_loss_weight 1.0
        --sre_temperature 2.0
        --sre_skew_loss_weight 1.0
        --sre_skew_lambda 0.01
        --projector_lr 5e-4
      )
      ;;
    dwa_kd)
      method_args=(
        --kd_loss_type dwa_kd
        --projector_config_path "${PROJECT_DIR}/config/dwa_kd_projectors.json"
        --projector_lr 5e-5
        --kd_objective forward_kl
        --ce_rate 0.5
        --kd_rate 0.5
        --dtw_rate 0.1
        --kd_temperature 2.0
        --teacher_temperature 1.0
        --kd_warmup_steps 300
        --dtw_gamma 2.0
        --dtw_gamma_start 2.0
        --dtw_gamma_end 0.8
        --dtw_gamma_steps 3570
        --dtw_band_width 5
        --dtw_band_source cma
      )
      ;;
    dskd_v2)
      method_args=(
        --kd_loss_type dskd_v2
        --projector_lr 5e-5
        --kd_objective forward_kl
        --kd_rate 0.5
        --kd_temperature 1.0
        --teacher_temperature 1.0
        --init_t2s_projector true
        --init_s2t_projector true
        --topk_vocab -1
        --t2s_agreement 1.0
      )
      ;;
    mcw_kd)
      method_args=(
        --kd_loss_type mcw_kd
        --proj_dim 512
        --projector_config_path "${PROJECT_DIR}/config/mcw_kd_projectors.json"
        --projector_lr 5e-5
        --kd_objective forward_kl
        --ce_rate 1.0
        --kd_rate 5.0
        --kd_temperature 1.0
        --teacher_temperature 1.0
        --top_k_vocab 400
        --mcw_tau_seq 2.0
        --mcw_window_size 4
        --mcw_ot_logits_rate 1.0
        --mcw_ot_hidden_rate 1.0
        --mcw_sinkhorn_alpha 0.1
        --mcw_sinkhorn_iter 100
      )
      ;;
    scva)
      method_args=(
        --kd_loss_type scva
        --scva_alpha 0.5
        --scva_weight 1.0
        --scva_n_layer_pairs 4
        --scva_layer_low_pct 0.4
        --scva_layer_high_pct 0.7
        --scva_min_vision_tokens 4
      )
      ;;
    scva_cgkd)
      method_args=(
        --kd_loss_type scva_cgkd
        --scva_min_vision_tokens 4
        --cgkd_temperature 1.0
        --scva_cgkd_ce_weight 0.7
        --scva_cgkd_lambda_v 0.3
        --scva_cgkd_lambda_g 0.3
      )
      ;;
    scva_sic_low)
      method_args=(
        --kd_loss_type scva_sic
        --scva_n_layer_pairs 4
        --scva_layer_low_pct 0.0
        --scva_layer_high_pct 0.4
        --scva_min_vision_tokens 4
        --scva_spatial_weight 0.1
        --scva_dbscan_min_samples 8
        --scva_dbscan_eps_percentile 3.0
        --scva_sic_ce_weight 1.0
        --scva_sic_lambda_v 0.3
        --scva_sic_lambda_sic 0.3
        --sic_num_regions 2
        --sic_max_clusters 0
        --sic_intervention_batch_size 2
        --sic_use_projector true
      )
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 2
      ;;
  esac

  if [[ "${method}" == "scva_sic_low" ]]; then
    method_train_env="${SCVA_SIC_TRAIN_ENV}"
  else
    method_train_env="${BASELINE_TRAIN_ENV}"
  fi

  (
    # shellcheck disable=SC1090
    source "${method_train_env}"
    if ! command -v python >/dev/null 2>&1; then
      echo "python was not found after activating ${method_train_env}" >&2
      exit 1
    fi
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "torchrun was not found after activating ${method_train_env}" >&2
      exit 1
    fi

    echo "[$(date --iso-8601=seconds)] Benchmarking ${method} on physical GPUs ${CUDA_VISIBLE_DEVICES}"
    echo "Training environment: ${VIRTUAL_ENV:-${CONDA_PREFIX:-unknown}}"
    echo "Python: $(command -v python)"
    echo "Torchrun: $(command -v torchrun)"

    torchrun \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_port "${MASTER_PORT}" \
      "${TRAIN_PY}" \
      "${common_args[@]}" \
      --output_dir "${output_dir}" \
      --benchmark_method "${method}" \
      "${method_args[@]}"
  ) 2>&1 | tee "${log_file}"

  echo
  echo "Result for ${method}:"
  if command -v column >/dev/null 2>&1; then
    # Show the common header and the row just appended by this method.
    { head -n 1 "${RESULTS_FILE}"; tail -n 1 "${RESULTS_FILE}"; } | column -t -s $'\t'
  else
    head -n 1 "${RESULTS_FILE}"
    tail -n 1 "${RESULTS_FILE}"
  fi
  echo
done

echo
echo "======================================================================"
echo "FINAL BENCHMARK SUMMARY (all methods completed)"
echo "======================================================================"

# Produce a compact comparison table from the detailed result file. Keeping it
# as TSV makes the summary easy to consume from spreadsheets or later scripts.
awk -F $'\t' 'BEGIN { OFS="\t" }
  NR == 1 {
    print "method", "avg_batch_ms", "avg_optimizer_step_ms", \
          "peak_allocated_gib", "peak_reserved_gib", "measured_steps";
    next
  }
  {
    print $1, $8, $7, $9, $10, $6
  }
' "${RESULTS_FILE}" > "${SUMMARY_FILE}"

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${SUMMARY_FILE}"
else
  cat "${SUMMARY_FILE}"
fi

fastest_method="$({ awk -F $'\t' '
  NR > 1 && ($8 + 0) < fastest {
    fastest = $8 + 0;
    method = $1
  }
  NR == 2 {
    fastest = $8 + 0;
    method = $1
  }
  END {
    if (method != "") printf "%s\t%.3f", method, fastest
  }
' "${RESULTS_FILE}"; })"

if [[ -n "${fastest_method}" ]]; then
  fastest_name="${fastest_method%%$'\t'*}"
  fastest_time="${fastest_method#*$'\t'}"
  echo
  echo "Fastest method by avg_batch_ms: ${fastest_name} (${fastest_time} ms/batch)"
fi

echo
echo "Detailed results: ${RESULTS_FILE}"
echo "Compact summary: ${SUMMARY_FILE}"
echo "Per-method logs: ${BENCHMARK_ROOT}/<method>.log"
