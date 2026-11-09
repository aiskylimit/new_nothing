#!/usr/bin/env bash
# Chuỗi tuần tự CHỈ trên GPU1: merge TIES -> eval merge -> phase2 distill -> eval student
# Trên 3 expert phase1_svgd_func_keystep_k1 (đã PASS cổng chặn V_i: mean V=0.0472, 0.6% V≈0)
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
PY=${PY:-python}
MODEL=${MODEL:-models/Qwen2.5-7B-Instruct}
TAG=svgd_func_keystep_k1
ROOT=outputs/proposal_v2/phase1_${TAG}
MD=outputs/proposal_v2/merge_${TAG}_ties
DD=outputs/proposal_v2/distill_${TAG}_ties
WDIR=outputs/proposal_v2/weights_${TAG}
LOG=logs/proposal_v2; mkdir -p "$LOG"
ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] $*"; }

log "=== 1. Merge TIES ==="
$PY -u -m proposal.merge --phase1_dir "$ROOT" --M 3 --strategy ties \
  --out_rank 16 --density 0.2 --out_dir "$MD" --device cuda \
  > "$LOG/merge_${TAG}_ties.log" 2>&1
if [ $? -ne 0 ]; then log "MERGE FAIL"; exit 1; fi
log "merge xong"

log "=== 2. Eval merge (chưa distill) ==="
$PY -u -m eval.run_eval --method proposal_v2_${TAG}_ties \
  --model_name "$MODEL" \
  --adapter "$MD/merged" --engine vllm --gpu_memory_utilization 0.4 \
  > "$LOG/eval_${TAG}_ties.log" 2>&1
if [ $? -ne 0 ]; then log "EVAL MERGE FAIL"; exit 1; fi
log "eval merge xong"

log "=== 3. Phase 2 distill ==="
$PY -u -m train.phase2_distill --model_name "$MODEL" \
  --train_file data/train_s1k_prefixed.jsonl \
  --weights_dir "$WDIR" \
  --teacher_adapter "$MD/merged" \
  --lambda_kl 1.0 --cutoff_len 16384 \
  --out_dir "$DD" --device cuda \
  > "$LOG/phase2_distill_${TAG}_ties.log" 2>&1
if [ $? -ne 0 ]; then log "DISTILL FAIL"; exit 1; fi
log "distill xong"

log "=== 4. Eval student cuối cùng ==="
$PY -u -m eval.run_eval --method proposal_v2_${TAG}_ties_distill \
  --model_name "$MODEL" \
  --adapter "$DD/student" --engine vllm --gpu_memory_utilization 0.4 \
  > "$LOG/eval_${TAG}_ties_distill.log" 2>&1
if [ $? -ne 0 ]; then log "EVAL STUDENT FAIL"; exit 1; fi
log "=== TOÀN BỘ CHUỖI XONG ==="
