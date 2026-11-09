#!/usr/bin/env bash
# Metric bảng ablation Phase 1: với mỗi kịch bản -> (1) js-key / js-nonkey / NLL trên tập proxy,
# (2) eval 3 expert trên 4 benchmark theo setting P-ALIGN chính xác (n=3, max_tokens = max_model_len = 4096).
#   GPU=<id> MODEL=7b|8b [SCENARIOS="full noDropout ..."] [WAIT_PATTERN=<pgrep -f>] bash scripts/run_ablation_metrics.sh
# Idempotent: bỏ qua file kết quả đã có. Lỗi thì ghi log rồi chuyển kịch bản tiếp theo.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:?can GPU=<id>}" HF_HUB_OFFLINE=1
PY=${PY:-python}
LOG=logs/ablation_metrics; mkdir -p "$LOG" results/ablation_metrics
ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] $*"; }
SCENARIOS="${SCENARIOS:-full noDropout noSVGD noOrtho onlySVGD onlyOrtho onlyDropout}"

case "${MODEL:?can MODEL=7b|8b}" in
  7b) MN=models/Qwen2.5-7B-Instruct; KS=outputs/proposal_v2/keystep_proxy500_10pct
      dir_of(){ if [ "$1" = full ]; then echo outputs/proposal_v2/phase1_svgd_func_keystep_k1; else echo "outputs/proposal_v2/phase1_svgd_func_keystep_k1_$1"; fi; } ;;
  8b) MN=models/Qwen3-8B; KS=outputs/proposal_v3/keystep_proxy500_10pct_qwen3
      dir_of(){ if [ "$1" = full ]; then echo outputs/proposal_v3/phase1_best_qwen3_8b; else echo "outputs/proposal_v3/phase1_qwen3_8b_$1_ablation"; fi; } ;;
  *) echo "MODEL phai la 7b hoac 8b"; exit 2 ;;
esac

if [ -n "${WAIT_PATTERN:-}" ]; then
  log "GPU$GPU: cho '$WAIT_PATTERN' xong..."; while pgrep -f "$WAIT_PATTERN" > /dev/null; do sleep 30; done; sleep 15
fi
log "cho key step proxy ($KS) du 500 mau..."
while [ "$(ls "$KS" 2>/dev/null | wc -l)" -lt 500 ]; do sleep 30; done; sleep 10

for sc in $SCENARIOS; do
  D=$(dir_of "$sc"); T="abl_${MODEL}_${sc}"
  NE=$(ls -d "$D"/expert_[0-9]* 2>/dev/null | wc -l)   # so expert thuc te (M=3 cac kich ban thuong, M=1 cho allOff 8b)
  [ "$NE" -ge 1 ] && [ -d "$D/expert_$((NE-1))" ] || { log "BO QUA $sc: chua co expert trong $D"; continue; }
  if [ "$NE" -ge 2 ] && [ ! -f "results/ablation_metrics/${T}_js.json" ]; then
    log "GPU$GPU: $T js"
    $PY -u -m proposal_v2.ablation_metrics --model_name "$MN" --phase1_dir "$D" --keystep_dir "$KS" \
      --out "results/ablation_metrics/${T}_js.json" > "$LOG/${T}_js.log" 2>&1 || log "JS THAT BAI $T (xem $LOG/${T}_js.log)"
  fi
  for i in $(seq 0 $((NE-1))); do
    M="${T}_expert${i}_exact"
    [ -f "results/${M}.json" ] && continue
    for attempt in 1 2; do
      log "GPU$GPU: eval $M (lan $attempt)"
      $PY -u -m eval.run_eval --method "$M" --model_name "$MN" --adapter "$D/expert_${i}" \
        --benchmarks math500 aime24 aime25 amc --k 3 --engine vllm --max_tokens 4096 --max_model_len 4096 \
        --gpu_memory_utilization "${VLLM_MEM:-0.45}" > "$LOG/${M}.log" 2>&1 && break
      log "EVAL THAT BAI $M lan $attempt"; sleep 30
    done
  done
  log "xong kich ban $sc"
done
log "GPU$GPU: HET ($MODEL)"
