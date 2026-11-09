#!/usr/bin/env bash
# Huấn luyện Phase 2 Distillation KHÔNG DÙNG KL (lambda_kl = 0.0)
# Kịch bản đỉnh cao 7B: proposal_v2_svgd_func_keystep_k1
# Chạy trên GPU 0

set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
PY=${PY:-python}
MODEL=${MODEL:-models/Qwen2.5-7B-Instruct}
TAG=svgd_func_keystep_k1
MD=outputs/proposal_v2/merge_${TAG}_ties
DD=outputs/proposal_v2/distill_${TAG}_ties_nokl
WDIR=outputs/proposal_v2/weights_${TAG}
LOG=logs/proposal_v2
mkdir -p "$LOG" "$DD"

ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] $*"; }

log "=== BẮT ĐẦU: Phase 2 Distill NO-KL (lambda_kl=0.0) ==="
log "Teacher Adapter (init student): $MD/merged"
log "Weights Dir: $WDIR"
log "Output Dir: $DD"

$PY -u -m train.phase2_distill \
  --model_name "$MODEL" \
  --train_file data/train_s1k_prefixed.jsonl \
  --weights_dir "$WDIR" \
  --teacher_adapter "$MD/merged" \
  --lambda_kl 0.0 \
  --cutoff_len 16384 \
  --epochs 3 \
  --lr 0.0001 \
  --global_batch 32 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --segmentation_mode paragraph \
  --loss_reduction step_mean \
  --seed 42 \
  --out_dir "$DD" \
  --device cuda \
  > "$LOG/phase2_distill_${TAG}_ties_nokl.log" 2>&1

if [ $? -ne 0 ]; then
  log "DISTILL NO-KL THẤT BẠI! Xem chi tiết tại $LOG/phase2_distill_${TAG}_ties_nokl.log"
  exit 1
fi
log "Huấn luyện Distill NO-KL thành công! Checkpoint tại $DD/student"

log "=== BẮT ĐẦU ĐÁNH GIÁ: Fast vLLM Eval trên 4 benchmark ==="
$PY -u -m eval.run_eval \
  --method proposal_v2_${TAG}_ties_distill_nokl \
  --model_name "$MODEL" \
  --adapter "$DD/student" \
  --engine vllm \
  --gpu_memory_utilization 0.4 \
  > "$LOG/eval_${TAG}_ties_distill_nokl.log" 2>&1

if [ $? -ne 0 ]; then
  log "EVAL THẤT BẠI! Xem chi tiết tại $LOG/eval_${TAG}_ties_distill_nokl.log"
  exit 1
fi

log "=== HOÀN TẤT TOÀN BỘ CHUỖI NO-KL ==="
log "Kết quả đã lưu tại results/proposal_v2_${TAG}_ties_distill_nokl.json"
