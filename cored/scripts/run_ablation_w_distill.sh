#!/bin/bash
# Ablation trọng số w_i (Phase 2) trên kịch bản 37.13 (expert phase1_svgd_func_keystep_k1, teacher TIES).
# A1 uniform | B1 chỉ U (1,0) | B2 chỉ rho (0,1) | B2' chỉ V thô (0,1). B0 (U+rho, 0.5/0.5) = 37.13 đã có.
# Mỗi cấu hình: distill (tham số y hệt B0) -> eval student. Chờ job merge sweep xong rồi mới chạy.
# Tiến trình hiện trong nvidia-smi với tiền tố hieunq10-.
#
# Dùng:  [ONLY="A1 B1"] bash scripts/run_ablation_w_distill.sh <GPU_ID> [PID_CHỜ]   (ONLY: chỉ chạy các cấu hình này)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="${1:?Usage: run_ablation_w_distill.sh <GPU_ID> [PID_CHO]}"
WAIT_PID="${2:-}"
export PROC_PREFIX=hieunq10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=${PY:-python}
TAG=svgd_func_keystep_k1
MODEL=${MODEL:-models/Qwen2.5-7B-Instruct}
TEACHER=outputs/proposal_v2/merge_${TAG}_ties/merged
WBASE=outputs/proposal_v2/weights_${TAG}
LOG=logs/proposal_v2/ablation_w; mkdir -p "$LOG"
ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] $*"; }

if [ -n "$WAIT_PID" ]; then
  log "chờ PID $WAIT_PID (merge sweep) kết thúc"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

# tên : weights_dir : cờ thêm
CONFIGS=(
  "A1:${WBASE}:--uniform_weights"
  "B1:${WBASE}_B1:"
  "B2:${WBASE}_B2:"
  "B2p:${WBASE}_B2p:"
)
for C in "${CONFIGS[@]}"; do
  IFS=: read -r NAME WD EXTRA <<<"$C"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $NAME "* ]]; then continue; fi
  DD=outputs/proposal_v2/distill_${TAG}_ablw_${NAME}
  METHOD=proposal_v2_${TAG}_ablw_${NAME}_distill
  if [ ! -d "$DD/student" ]; then
    log "distill $NAME (weights=$WD $EXTRA)"
    $PY -u scripts/titled_run.py "distill_${NAME}" train.phase2_distill --model_name "$MODEL" \
      --train_file data/train_s1k_prefixed.jsonl --weights_dir "$WD" --teacher_adapter "$TEACHER" \
      --lambda_kl 1.0 --cutoff_len 16384 --out_dir "$DD" --device cuda $EXTRA \
      > "$LOG/distill_${NAME}.log" 2>&1 || { log "DISTILL $NAME FAIL"; continue; }
  fi
  if [ ! -f "results/${METHOD}.json" ]; then
    log "eval student $NAME"
    $PY -u -m eval.run_eval --method "$METHOD" --adapter "$DD/student" --model_name "$MODEL" \
      --engine vllm --gpu_memory_utilization 0.4 > "$LOG/eval_${NAME}.log" 2>&1 || log "EVAL $NAME FAIL"
  fi
done

log "=== TỔNG HỢP ablation w_i (B0 = proposal_v2_${TAG}_ties_distill) ==="
$PY - <<PYEOF
import json, os
rows = [("B0 U+rho", "proposal_v2_${TAG}_ties_distill")] + [(n, f"proposal_v2_${TAG}_ablw_{n}_distill") for n in ["A1","B1","B2","B2p"]]
for n, m in rows:
    p = f"results/{m}.json"
    if not os.path.exists(p): print(f"  {n:9s} (chưa có)"); continue
    d = json.load(open(p)); b = d["benchmarks"]
    per = " ".join(f"{k}={v['pass@1']*100:.1f}/{v['pass@3']*100:.1f}" for k, v in b.items())
    print(f"  {n:9s} avgP@1={d['avg']*100:5.2f} avgP@3={d.get('avg_passk',0)*100:5.2f} tok={d['overall_tokens']['mean']:.0f} | {per}")
PYEOF
