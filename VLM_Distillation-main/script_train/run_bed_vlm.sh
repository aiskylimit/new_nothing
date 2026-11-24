#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash script_train/run_bed_vlm.sh PAIR METHOD [--dry-run] [-- TRAIN_ARGS...]

Pairs: fastvlm, qwen2
Main methods: bed_vlm, sft, sra, mcw_kd, em_kd, dwa_kd, dskd_v2
FastVLM ablations: low, middle, high, scva_only, sic_only, dbscan_{4,8,12,16}
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage >&2
  exit 2
fi

pair=$1
method=$2
shift 2
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
  dry_run=true
  shift
fi
if [[ ${1:-} == -- ]]; then
  shift
elif (( $# )); then
  usage >&2
  exit 2
fi

cd -- "${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
case "$pair" in
  fastvlm)
    student=${STUDENT_MODEL:-KamilaMila/FastVLM-0.5B}
    teacher=${TEACHER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}
    ;;
  qwen2)
    student=${STUDENT_MODEL:-Qwen/Qwen2-VL-2B-Instruct}
    teacher=${TEACHER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
    ;;
  *) usage >&2; exit 2 ;;
esac

seed=${SEED:-1337}
fraction=1.0
subset_args=()
teacher_args=(--teacher_model_name "$teacher")
criterion_args=()
case "$method" in
  sft)
    teacher_args=()
    ;;
  sra)
    criterion_args=(--kd_loss_type sra --sre_use_projector true
      --teacher_layer_mapping -3 -1 --student_layer_mapping -3 -1
      --sre_alpha 0.5 --sre_p 1.0 --sre_span_loss_weight 1.0 --sre_geom_loss_weight 3.0
      --sre_logit_loss_weight 1.0 --sre_temperature 2.0 --sre_skew_loss_weight 1.0
      --sre_skew_lambda 0.01 --projector_lr 5e-4)
    ;;
  em_kd)
    criterion_args=(--kd_loss_type emkd --em_kd_alpha 0.5 --em_kd_beta 0.25
      --em_kd_gamma 25.0 --em_kd_temperature 1.0
      --em_kd_max_vision_tokens 512 --em_kd_max_text_tokens 1024)
    ;;
  mcw_kd)
    criterion_args=(--kd_loss_type mcw_kd --projector_config_path config/mcw_kd_projectors.json
      --projector_lr 5e-5 --kd_objective forward_kl --ce_rate 1.0 --kd_rate 5.0
      --kd_temperature 1.0 --mcw_tau_seq 2.0 --mcw_window_size 4 --mcw_ot_logits_rate 1.0
      --mcw_ot_hidden_rate 1.0 --mcw_sinkhorn_alpha 0.1 --mcw_sinkhorn_iter 100)
    ;;
  dwa_kd)
    weight=1.0
    temperature=1.0
    if [[ "$pair" == fastvlm ]]; then
      weight=0.5
      temperature=2.0
    fi
    criterion_args=(--kd_loss_type dwa_kd --projector_config_path config/dwa_kd_projectors.json
      --projector_lr 5e-5 --kd_objective forward_kl --ce_rate "$weight" --kd_rate "$weight"
      --kd_temperature "$temperature" --dtw_rate 0.1 --kd_warmup_steps 300 --dtw_gamma 2.0
      --dtw_gamma_start 2.0 --dtw_gamma_end 0.8 --dtw_gamma_steps 3570 --dtw_band_width 5
      --dtw_band_source cma)
    ;;
  dskd_v2)
    criterion_args=(--kd_loss_type dskd_v2 --projector_lr 5e-5 --kd_objective forward_kl
      --kd_rate 0.5 --kd_temperature 1.0 --teacher_temperature 1.0
      --init_t2s_projector true --init_s2t_projector true --topk_vocab -1 --t2s_agreement 1.0)
    ;;
  bed_vlm|low|middle|high|scva_only|sic_only|dbscan_4|dbscan_8|dbscan_12|dbscan_16)
    if [[ "$method" != bed_vlm ]]; then
      if [[ "$pair" != fastvlm ]]; then
        echo "The paper ablations use the fastvlm pair." >&2
        exit 2
      fi
      fraction=0.3
      subset_args=(--data_subset_seed "${SUBSET_SEED:-1337}")
    fi
    low=0.0
    high=0.4
    lambda_v=${LAMBDA_V:-0.3}
    lambda_s=${LAMBDA_S:-0.3}
    min_samples=12
    case "$method" in
      middle) low=0.4; high=0.7 ;;
      high) low=0.7; high=1.0 ;;
      scva_only) lambda_s=0.0 ;;
      sic_only) lambda_v=0.0 ;;
      dbscan_*) min_samples=${method#dbscan_} ;;
    esac
    criterion_args=(--kd_loss_type bed_vlm --scva_sic_lambda_v "$lambda_v"
      --scva_sic_lambda_sic "$lambda_s" --scva_layer_low_pct "$low" --scva_layer_high_pct "$high"
      --scva_dbscan_min_samples "$min_samples" --sic_num_regions "${SIC_NUM_REGIONS:-16}")
    ;;
  *) usage >&2; exit 2 ;;
esac

command=(torchrun --nproc_per_node "${NPROC_PER_NODE:-gpu}" --master_port "${MASTER_PORT:-29501}" train.py
  --model_name "$student" "${teacher_args[@]}"
  --data_path "${DATA_PATH:-train_data/llava_v1_5_mix665k.json}" --image_dir "${IMAGE_DIR:-train_data}"
  --output_dir "${OUTPUT_DIR:-outputs/bed_vlm/${pair}/${method}/seed-${seed}}"
  --percent_data "$fraction" "${subset_args[@]}" --seed "$seed"
  --lora true --lora_r 128 --lora_alpha 256 --lora_dropout 0.05
  --per_device_train_batch_size "${PER_DEVICE_BS:-1}" --gradient_accumulation_steps "${GRAD_ACCUM:-8}"
  --num_train_epochs 1 --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03
  --lr_scheduler_type cosine --bf16 true --save_strategy epoch --save_total_limit 2
  --logging_steps 50 --dataloader_num_workers "${DATALOADER_WORKERS:-2}"
  --max_len 2048 --image_resolution low "${criterion_args[@]}" "$@")

if "$dry_run"; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
