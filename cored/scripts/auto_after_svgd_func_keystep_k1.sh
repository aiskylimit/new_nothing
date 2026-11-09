#!/usr/bin/env bash
# Sau khi training phase1_svgd_func_keystep_k1 (PID train hiện tại: 3291131, GPU0) xong:
#   B1. proposal_v2.keystep_diversity_report  — so sánh V (đa dạng) key-step vs step thường
#       bằng U/V của CHÍNH 3 expert đã train (khác keystep_select.py: cái đó dùng entropy
#       BASE MODEL để CHỌN key step trước khi train).
#   B2. eval.run_eval cho từng expert riêng lẻ (expert_0, expert_1, expert_2).
#   B3. proposal.merge (chỉ chiến lược ties, tốt nhất theo các sweep trước) rồi eval bản merge.
#   B4. train.compute_weights                 — cổng chặn V_i chuẩn (§13 bước 6) trên TOÀN
#       BỘ step, để so với bản không-keystep (phase1_svgd_func_k1, đã FAIL 98.4% V≈0).
#   B5. In bảng tổng hợp accuracy.
#
# Toàn bộ chạy trên GPU0 (chờ train nhả GPU0 xong mới bắt đầu).
# Dùng:  bash scripts/auto_after_svgd_func_keystep_k1.sh [TRAIN_PID]
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=${PY:-python}
MODEL=${MODEL:-models/Qwen2.5-7B-Instruct}
TAG=svgd_func_keystep_k1
ROOT=outputs/proposal_v2/phase1_${TAG}
KEYDIR=outputs/proposal_v2/keystep_s1k_prefixed_10pct
MERGE_STRATEGY=ties
LOG=logs/proposal_v2; mkdir -p "$LOG"
TRAIN_PID="${1:-3291131}"
ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] $*"; }

log "chờ training PID ${TRAIN_PID} (phase1_${TAG}, GPU0) xong ..."
while ps -p "$TRAIN_PID" > /dev/null 2>&1; do sleep 30; done
log "=== training xong, GPU0 rảnh, bắt đầu B1 ==="

# --- B1. Diversity report: V trên key step vs step thường (dùng expert đã train) ---
if [ ! -f "$ROOT/keystep_diversity_summary.json" ]; then
  $PY -u -m proposal_v2.keystep_diversity_report \
    --model_name "$MODEL" --phase1_dir "$ROOT" --keystep_dir "$KEYDIR" \
    --train_file data/train_s1k_prefixed.jsonl --cutoff_len 16384 --device cuda \
    2>&1 | tee "$LOG/keystep_diversity_${TAG}.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { log "B1 FAIL"; exit 1; }
else
  log "B1 đã có -> bỏ qua"
fi

# --- B2. Eval 3 expert riêng lẻ ---
for i in 0 1 2; do
  if [ ! -f "results/proposal_v2_${TAG}_expert${i}.json" ]; then
    log "=== B2. eval expert_${i} ==="
    $PY -u -m eval.run_eval --method "proposal_v2_${TAG}_expert${i}" \
      --adapter "$ROOT/expert_${i}" --model_name "$MODEL" \
      --engine vllm --gpu_memory_utilization 0.85 \
      2>&1 | tee "$LOG/eval_${TAG}_expert${i}.log"
  else
    log "B2 expert_${i} đã có -> bỏ qua"
  fi
done

# --- B3. Merge (chỉ ties) + eval ---
MD="outputs/proposal_v2/merge_${TAG}_${MERGE_STRATEGY}"
if [ ! -f "$MD/merged/adapter_model.safetensors" ]; then
  log "=== B3. merge strategy=$MERGE_STRATEGY ==="
  $PY -u -m proposal.merge --phase1_dir "$ROOT" --M 3 --strategy "$MERGE_STRATEGY" \
    --out_rank 16 --density 0.2 --out_dir "$MD" \
    2>&1 | tee "$LOG/merge_${TAG}_${MERGE_STRATEGY}.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { log "B3 merge FAIL"; exit 1; }
fi
if [ ! -f "results/proposal_v2_${TAG}_${MERGE_STRATEGY}.json" ]; then
  log "=== B3. eval merge $MERGE_STRATEGY ==="
  $PY -u -m eval.run_eval --method "proposal_v2_${TAG}_${MERGE_STRATEGY}" \
    --adapter "$MD/merged" --model_name "$MODEL" \
    --engine vllm --gpu_memory_utilization 0.85 \
    2>&1 | tee "$LOG/eval_${TAG}_${MERGE_STRATEGY}.log"
fi

# --- B4. Cổng chặn V_i chuẩn, TOÀN BỘ step (đối chiếu bản không-keystep) ---
WDIR="outputs/proposal_v2/weights_${TAG}"
if [ "$(ls "$WDIR"/*.npz 2>/dev/null | wc -l)" -lt 900 ]; then
  log "=== B4. compute_weights (toàn bộ step) ==="
  $PY -u -m train.compute_weights --model_name "$MODEL" --train_file data/train_s1k_prefixed.jsonl \
    --phase1_dir "$ROOT" --weights_dir "$WDIR" --M 3 \
    --disagreement_mode normalized --cutoff_len 16384 --chunk_size 128 --preview 3 \
    2>&1 | tee "$LOG/compute_weights_${TAG}.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { log "B4 FAIL"; exit 1; }
else
  log "B4 đã có -> bỏ qua"
fi

# --- B5. Tổng hợp ---
log "=== TỔNG HỢP ${TAG} ==="
$PY - <<PYEOF
import json, os
print("-- experts --")
for i in range(3):
    p = f"results/proposal_v2_${TAG}_expert{i}.json"
    if not os.path.exists(p):
        print(f"  expert_{i} (chưa có)"); continue
    d = json.load(open(p))
    print(f"  expert_{i}: avgP@1={d['avg']*100:.2f}")
print("-- merge --")
p = f"results/proposal_v2_${TAG}_${MERGE_STRATEGY}.json"
if os.path.exists(p):
    d = json.load(open(p))
    print(f"  ${MERGE_STRATEGY}: avgP@1={d['avg']*100:.2f}")
else:
    print("  ${MERGE_STRATEGY} (chưa có)")
ks = "outputs/proposal_v2/phase1_${TAG}/keystep_diversity_summary.json"
if os.path.exists(ks):
    d = json.load(open(ks))
    k, o = d["key_step"]["V"], d["non_key_step"]["V"]
    print(f"-- diversity (V) -- key_step mean={k.get('mean'):.4f} vs non_key mean={o.get('mean'):.4f}")
PYEOF
log "=== XONG ==="
