#!/bin/bash
# Bổ sung các cách merge (ta, iso_c, tsv, ctm) cho kịch bản tốt nhất sau 2 phase:
#   Phase 1 experts = outputs/proposal_v2/phase1_svgd_func_keystep_k1
#   merge TIES + distill (Phase 2) => proposal_v2_svgd_func_keystep_k1_ties_distill, Avg P@1 = 37.13
# Mỗi strategy chạy đúng chuỗi như run_merge_distill_keystep.sh:
#   merge -> eval merge -> Phase 2 distill (lambda_kl=1, cutoff 16384, weights_dir cũ) -> eval student
# Giai đoạn A (nhanh): merge + eval merge cho cả 4 strategy. Giai đoạn B (~2.5h/strategy): distill + eval student.
#
# Dùng:  bash scripts/run_keystep_merge_sweep.sh <GPU_ID>
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="${1:?Usage: run_keystep_merge_sweep.sh <GPU_ID>}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=${PY:-python}
TAG=svgd_func_keystep_k1
ROOT=outputs/proposal_v2/phase1_${TAG}
WDIR=outputs/proposal_v2/weights_${TAG}
LOG=logs/proposal_v2/keystep_merge_sweep; mkdir -p "$LOG"
MODEL=${MODEL:-models/Qwen2.5-7B-Instruct}
STRATS=(ta iso_c tsv ctm)
ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] $*"; }

log "=== A. merge + eval merge (chưa distill) ==="
for S in "${STRATS[@]}"; do
  MD=outputs/proposal_v2/merge_${TAG}_${S}
  if [ ! -f "$MD/merged/adapter_model.safetensors" ]; then
    log "merge $S"
    $PY -u -m proposal.merge --phase1_dir "$ROOT" --M 3 --strategy "$S" \
      --out_rank 16 --density 0.2 --per_task_rank 16 --out_dir "$MD" --device cuda \
      > "$LOG/merge_${S}.log" 2>&1 || { log "MERGE $S FAIL"; continue; }
  fi
  if [ ! -f "results/proposal_v2_${TAG}_${S}.json" ]; then
    log "eval merge $S"
    $PY -u -m eval.run_eval --method "proposal_v2_${TAG}_${S}" --adapter "$MD/merged" --model_name "$MODEL" \
      --engine vllm --gpu_memory_utilization 0.4 \
      > "$LOG/eval_merge_${S}.log" 2>&1 || log "EVAL MERGE $S FAIL"
  fi
done

log "=== B. Phase 2 distill + eval student ==="
for S in "${STRATS[@]}"; do
  MD=outputs/proposal_v2/merge_${TAG}_${S}
  DD=outputs/proposal_v2/distill_${TAG}_${S}
  if [ ! -f "$MD/merged/adapter_model.safetensors" ]; then log "thiếu merge $S -> bỏ qua"; continue; fi
  if [ ! -d "$DD/student" ]; then
    log "distill $S"
    $PY -u -m train.phase2_distill --model_name "$MODEL" \
      --train_file data/train_s1k_prefixed.jsonl \
      --weights_dir "$WDIR" --teacher_adapter "$MD/merged" \
      --lambda_kl 1.0 --cutoff_len 16384 --out_dir "$DD" --device cuda \
      > "$LOG/distill_${S}.log" 2>&1 || { log "DISTILL $S FAIL"; continue; }
  fi
  if [ ! -f "results/proposal_v2_${TAG}_${S}_distill.json" ]; then
    log "eval student $S"
    $PY -u -m eval.run_eval --method "proposal_v2_${TAG}_${S}_distill" --adapter "$DD/student" --model_name "$MODEL" \
      --engine vllm --gpu_memory_utilization 0.4 \
      > "$LOG/eval_student_${S}.log" 2>&1 || log "EVAL STUDENT $S FAIL"
  fi
done

log "=== TỔNG HỢP (mốc: TIES distill 37.13 | TIES merge chưa distill xem results/proposal_v2_${TAG}_ties.json) ==="
$PY - <<PYEOF
import json, os
tag = "${TAG}"
for s in ["ties"] + "${STRATS[*]}".split():
    for suf in ["", "_distill"]:
        p = f"results/proposal_v2_{tag}_{s}{suf}.json"
        if not os.path.exists(p):
            print(f"  {s+suf:16s} (chưa có)"); continue
        d = json.load(open(p)); b = d["benchmarks"]
        per = " ".join(f"{k}={v['pass@1']*100:.1f}" for k, v in b.items())
        print(f"  {s+suf:16s} avgP@1={d['avg']*100:5.2f} avgP@3={d.get('avg_passk',0)*100:5.2f} tok={d['overall_tokens']['mean']:.0f} | {per}")
PYEOF
