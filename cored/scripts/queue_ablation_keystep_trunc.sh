#!/bin/bash
# Hàng đợi ablation CoReD (tuần tự trên MỘT GPU). Mỗi dòng ~7-9h -> chọn dòng cần chạy qua ROWS.
#   Tỉ lệ key-step:  K0 (=noSVGD có sẵn) | K15 | K20 | K100        (10% = ours, đã có: 37.13)
#   Cắt CoT:         T100 (không cắt) | T025 | T050 | T075 | TRAND (ngẫu nhiên khớp k*)   (adaptive k* = ours)
#   K100 dùng lại Phase 1 (phase1_svgd_func_k1) + cache U,V (weights_svgd_func_k1_full) đã có -> chỉ còn merge+distill+eval (~4h)
# Dùng:  ROWS="K15 K20 K100" bash scripts/queue_ablation_keystep_trunc.sh <GPU_ID> [PID_CHỜ]
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:?Usage: queue_ablation_keystep_trunc.sh <GPU_ID> [PID_CHO]}"; WAIT_PID="${2:-}"
ROWS="${ROWS:-K100 K15 K20 K0 T100 T050 T025 T075 TRAND}"
PREF=data/train_s1k_prefixed.jsonl; TD=data/ablation_trunc
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
for R in $ROWS; do
  case "$R" in
    K0)    unset WEIGHTS_SRC; export PHASE1_DIR=outputs/proposal_v2/phase1_svgd_func_keystep_k1_noSVGD TRAIN_FILE=$PREF VMASK=outputs/proposal_v2/keystep_s1k_prefixed_10pct ;;  # V đo trên key-step 10% tham chiếu
    K15)   unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$PREF KEY_FRAC=0.15 ;;
    K20)   unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$PREF KEY_FRAC=0.20 ;;
    K100)  export PHASE1_DIR=outputs/proposal_v2/phase1_svgd_func_k1 WEIGHTS_SRC=outputs/proposal_v2/weights_svgd_func_k1_full TRAIN_FILE=$PREF VMASK=all ;;  # đã train sẵn: svgd_func trên MỌI step (không key-step)
    T100)  unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$TD/full.jsonl KEY_FRAC=0.10 ;;
    T025)  unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$TD/f025.jsonl KEY_FRAC=0.10 ;;
    T050)  unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$TD/f050.jsonl KEY_FRAC=0.10 ;;
    T075)  unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$TD/f075.jsonl KEY_FRAC=0.10 ;;
    TRAND) unset PHASE1_DIR WEIGHTS_SRC VMASK; export TRAIN_FILE=$TD/rand.jsonl KEY_FRAC=0.10 ;;
    *) echo "dòng lạ: $R"; continue ;;
  esac
  echo "[$(date '+%F %T')] === dòng $R ==="
  TAG=$R bash scripts/run_ablation_row.sh "$GPU" || echo "[$(date '+%F %T')] dòng $R lỗi -> tiếp dòng sau"
done
echo "[$(date '+%F %T')] === XONG hàng đợi: $ROWS ==="
