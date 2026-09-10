#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-${PIPELINE_MODE:-full851k}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${MODEL_DIR:?source env.sh first}"
: "${VAE_DIR:?source env.sh first}"
: "${DATA_DIR:?source env.sh first}"
: "${STREAM_MANIFEST:?source env.sh first}"
: "${VENV_DIR:?source env.sh first}"

IFS=',' read -r -a gpu_array <<< "${GPU_IDS:?GPU_IDS is required}"
actual_gpu_count="${#gpu_array[@]}"
if (( actual_gpu_count != NUM_GPUS )); then
  echo "GPU_IDS contains $actual_gpu_count devices but NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi
denominator=$((NUM_GPUS * TRAIN_BATCH_SIZE))
if (( EFFECTIVE_BATCH % denominator != 0 )); then
  echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH is not divisible by GPUs*micro_batch=$denominator" >&2
  exit 3
fi
GRADIENT_ACCUMULATION_STEPS=$((EFFECTIVE_BATCH / denominator))

case "$MODE" in
  pilot)
    DATASET_PAIRS="${DATASET_PAIRS:-128}"
    MAX_TRAIN_STEPS="${PILOT_TRAIN_STEPS:-2}"
    WARMUP_STEPS=0
    ;;
  full85k)
    DATASET_PAIRS="${DATASET_PAIRS:-85000}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1329}"
    WARMUP_STEPS="${WARMUP_STEPS:-200}"
    ;;
  full851k)
    DATASET_PAIRS="${DATASET_PAIRS:-851293}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-13302}"
    WARMUP_STEPS="${WARMUP_STEPS:-200}"
    ;;
  *) echo "usage: $0 [pilot|full85k|full851k]" >&2; exit 4 ;;
esac

RUN_NAME="${RUN_NAME:-q3_dspo_sdxl_${MODE}_eb${EFFECTIVE_BATCH}_${NUM_GPUS}gpu}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR" "$CACHE_DIR" "$STREAM_CACHE"

if [[ -f "$RUN_DIR/model_index.json" && -f "$RUN_DIR/exit-code.txt" ]] \
   && [[ "$(<"$RUN_DIR/exit-code.txt")" == 0 ]]; then
  echo "Training already completed; reusing $RUN_DIR"
  printf '%s\n' "$MAX_TRAIN_STEPS" > "$RUN_DIR/final-step.txt"
  exit 0
fi

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
elif find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
  resume_args+=(--resume_from_checkpoint=latest)
fi

launcher=("$VENV_DIR/bin/python" -m accelerate.commands.launch --num_machines=1)
if (( NUM_GPUS > 1 )); then
  launcher+=(--multi_gpu --num_processes="$NUM_GPUS" --main_process_port="${MASTER_PORT:-29500}")
else
  launcher+=(--num_processes=1)
fi
launcher+=(--mixed_precision=bf16)

command=(
  "${launcher[@]}" "$PROJECT_ROOT/train.py"
  --mixed_precision=bf16
  --pretrained_model_name_or_path="$MODEL_DIR"
  --pretrained_vae_model_name_or_path="$VAE_DIR"
  --sdxl --resolution=1024
  --dataset_name=parquet --train_data_dir="$DATA_DIR"
  --pickapic_streaming_manifest="$STREAM_MANIFEST"
  --pickapic_stream_cache="$STREAM_CACHE"
  --streaming_retries=1 --dataloader_num_workers=0 --pair_label_policy=error
  --max_train_samples="$DATASET_PAIRS" --seed="$SEED"
  --train_batch_size="$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
  --max_train_steps="$MAX_TRAIN_STEPS"
  --lr_scheduler=constant_with_warmup --lr_warmup_steps="$WARMUP_STEPS"
  --learning_rate="${BASE_LEARNING_RATE:-1e-8}" --scale_lr
  --gradient_checkpointing --allow_tf32
  --checkpointing_steps="$MAX_TRAIN_STEPS" --checkpoints_total_limit=1
  --no_hflip --proportion_empty_prompts=0
  --preference_loss=step_aware_tbpo
  --ratio_beta="${RATIO_BETA:-500}"
  --ratio_margin_gradient_scale="${RATIO_MARGIN_GRADIENT_SCALE:-2500}"
  --ratio_reduction=mean --transition_estimator=pointwise
  --state_correction=exact_gaussian --transition_variance_floor=1e-12
  --confidence_hidden_dim=128 --confidence_learning_rate=1e-4
  --confidence_timestep_embedding_dim=32 --confidence_loss_weight=1.0
  --confidence_min_policy_weight=0.05 --confidence_label_smoothing=0.05
  --confidence_policy_weight_normalization=global_microbatch_mean
  --reference_anchor_output_gradient_ratio=0.10
  --winner_anchor_output_gradient_ratio=0.25
  --bregman_lambda=0 --bregman_scale=4 --max_exp_argument=30
  --winner_anchor_type=dspo_score
  --dspo_probability_source=mse_preference --dspo_logit_beta=0.01
  --dspo_probability_temperature=1.0 --dspo_score_correction_scale=0.25
  --report_to=tensorboard --cache_dir="$CACHE_DIR/huggingface"
  --output_dir="$RUN_DIR" "${resume_args[@]}"
)

{
  printf 'mode=%s\nmodel_family=sdxl\nnum_gpus=%s\ngpu_ids=%s\n' "$MODE" "$NUM_GPUS" "$GPU_IDS"
  printf 'micro_batch=%s\naccumulation=%s\neffective_batch=%s\noptimizer_steps=%s\n' \
    "$TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$EFFECTIVE_BATCH" "$MAX_TRAIN_STEPS"
  printf 'offline_strict=%s\ncommand=' "$RATIO_OFFLINE_STRICT"
  printf '%q ' "${command[@]}"; printf '\n'
} > "$RUN_DIR/command.txt"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi

"$VENV_DIR/bin/python" -m pip freeze > "$RUN_DIR/pip-freeze.txt" 2>/dev/null || true
nvidia-smi -q > "$RUN_DIR/nvidia-smi-before.txt"
rm -f "$RUN_DIR/exit-code.txt"
set +e
CUDA_VISIBLE_DEVICES="$GPU_IDS" "${command[@]}" 2>&1 | tee -a "$RUN_DIR/console.log"
status=${PIPESTATUS[0]}
set -e
nvidia-smi -q > "$RUN_DIR/nvidia-smi-after.txt"
printf '%s\n' "$status" > "$RUN_DIR/exit-code.txt"
if (( status == 0 )); then
  printf '%s\n' "$MAX_TRAIN_STEPS" > "$RUN_DIR/final-step.txt"
fi
exit "$status"

