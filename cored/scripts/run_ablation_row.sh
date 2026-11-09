#!/bin/bash
# Một dòng ablation (CoReD) chạy TRỌN chuỗi tới điểm sau distill:
#   [keystep_select] -> Phase 1 (3 expert, SVGD-func + key-step) -> merge TIES -> eval merge
#   -> compute_weights (U,V,w) -> Phase 2 distill -> eval student (P@1/P@3 sau distill)
# Mọi siêu tham số y hệt kịch bản 37.13 (phase1_svgd_func_keystep_k1); CHỈ đổi thứ được ablate.
# Có cache: bước nào đã xong thì bỏ qua -> chạy lại an toàn. Tiến trình mang tiền tố hieunq10-.
#
# Biến môi trường (bắt buộc: TAG, TRAIN_FILE):
#   TAG         tên dòng, vd K15 / T050        (kết quả: proposal_v2_abl_<TAG>_{ties,distill})
#   TRAIN_FILE  dữ liệu train (cùng file cho keystep/phase1/weights/distill)
#   KEY_FRAC    tỉ lệ key-step có repulsion (mặc định 0.10). Bỏ qua nếu có PHASE1_DIR.
#   PHASE1_DIR  dùng luôn 3 expert Phase 1 có sẵn (vd dòng 0% = noSVGD) -> bỏ keystep_select+Phase 1
#   KEYDIR      dùng luôn thư mục key-step có sẵn (nếu không set thì tự sinh)
#   VMASK       thư mục key_mask để tính "V ở key step" (mặc định = key-step của dòng; "all" = mọi step)
#   SKIP_EVAL_MERGE  =1 (mặc định) bỏ eval merge; đặt 0 để eval merge
#   WEIGHTS_SRC cache U,V có sẵn của CHÍNH 3 expert đó -> bỏ compute_weights (4-6h), chỉ tính lại w
#               bằng rewrite_weights.py (normalized rho, lambda 0.5/0.5 — y hệt B0), vài giây trên CPU
# Dùng:  TAG=K15 TRAIN_FILE=data/train_s1k_prefixed.jsonl KEY_FRAC=0.15 bash scripts/run_ablation_row.sh <GPU_ID>
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="${1:?Usage: run_ablation_row.sh <GPU_ID>}"
: "${TAG:?cần TAG}"; : "${TRAIN_FILE:?cần TRAIN_FILE}"
KEY_FRAC="${KEY_FRAC:-0.10}"
export PROC_PREFIX=hieunq10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=${PY:-python}
MODEL=Qwen/Qwen2.5-7B-Instruct
BASE=outputs/proposal_v2/abl_${TAG}
LOG=logs/proposal_v2/abl_${TAG}; mkdir -p "$LOG" "$BASE"
TEMPLATE=configs/proposal_v2/phase1_svgd_func_keystep_k1.yaml
ts(){ date '+%F %T'; }; log(){ echo "[$(ts)] [$TAG] $*"; }
nnpz(){ ls "$1"/*.npz 2>/dev/null | wc -l; }
wait_free(){  # $1 = MiB cần trống trên GPU $CUDA_VISIBLE_DEVICES; đủ 2 lần đo liên tiếp cách 30s
  local need=$1 ok=0 t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 86400 ]; do
    local free; free=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    if [ "${free:-0}" -ge "$need" ]; then ok=$((ok+1)); [ "$ok" -ge 2 ] && return 0; else ok=0; fi
    sleep 30
  done
  return 1
}
run(){ local title=$1 mod=$2; shift 2; $PY -u scripts/titled_run.py "$title" "$mod" "$@"; }

# ---- 1. key-step + Phase 1 ----------------------------------------------------
if [ -n "${PHASE1_DIR:-}" ]; then
  P1=$PHASE1_DIR; log "dùng Phase 1 có sẵn: $P1"
else
  KD="${KEYDIR:-$BASE/keystep}"
  if [ "$(nnpz "$KD")" -lt 900 ]; then
    log "chờ GPU trống 36GB cho keystep_select"; wait_free 36000
    log "keystep_select frac=$KEY_FRAC"
    run "ks_${TAG}" proposal_v2.keystep_select --model_name "$MODEL" --train_file "$TRAIN_FILE" \
      --out_dir "$KD" --frac "$KEY_FRAC" --cutoff_len 16384 --device cuda > "$LOG/keystep_select.log" 2>&1 \
      || { log "KEYSTEP FAIL"; exit 1; }
  fi
  P1=$BASE/phase1
  if [ ! -f "$P1/expert_2/adapter_model.safetensors" ]; then
    mkdir -p configs/proposal_v2/gen
    CFG=configs/proposal_v2/gen/phase1_abl_${TAG}.yaml
    sed -e "s|^train_file:.*|train_file: ${TRAIN_FILE}|" -e "s|^out_dir:.*|out_dir: ${P1}|" \
        -e "s|^keystep_dir:.*|keystep_dir: ${KD}|" "$TEMPLATE" > "$CFG"
    log "chờ GPU trống 50GB cho Phase 1"; wait_free 50000
    log "Phase 1 (config $CFG)"
    run "p1_${TAG}" proposal_v2.phase1 --config "$CFG" --device cuda > "$LOG/phase1.log" 2>&1 \
      || { log "PHASE1 FAIL"; exit 1; }
  fi
fi

# ---- 2. merge TIES + eval merge ------------------------------------------------
MD=$BASE/merge_ties
if [ ! -f "$MD/merged/adapter_model.safetensors" ]; then
  log "chờ GPU trống 12GB cho merge"; wait_free 12000
  log "merge TIES"
  $PY -u -m proposal.merge --phase1_dir "$P1" --M 3 --strategy ties --out_rank 16 --density 0.2 \
    --out_dir "$MD" --device cuda > "$LOG/merge.log" 2>&1 || { log "MERGE FAIL"; exit 1; }
fi
if [ "${SKIP_EVAL_MERGE:-1}" != "1" ] && [ ! -f "results/proposal_v2_abl_${TAG}_ties.json" ]; then
  log "eval merge"
  $PY -u scripts/titled_run.py "ev_${TAG}_ties" eval.run_eval --method "proposal_v2_abl_${TAG}_ties" \
    --adapter "$MD/merged" --model_name "$MODEL" --engine vllm --gpu_memory_utilization 0.4 \
    > "$LOG/eval_merge.log" 2>&1 || log "EVAL MERGE FAIL (tiếp tục)"
fi

# ---- 3. compute_weights (U,V,w trên toàn bộ step) -------------------------------
WD=$BASE/weights
if [ "$(nnpz "$WD")" -lt 900 ] && [ -n "${WEIGHTS_SRC:-}" ]; then
  log "tính lại w từ cache U,V $WEIGHTS_SRC (normalized, 0.5/0.5)"
  CUDA_VISIBLE_DEVICES="" $PY scripts/rewrite_weights.py --weights_dir "$WEIGHTS_SRC" --out_dir "$WD" \
    --lambda_U 0.5 --lambda_D 0.5 --disagreement_mode normalized > "$LOG/rewrite_weights.log" 2>&1 \
    || { log "REWRITE WEIGHTS FAIL"; exit 1; }
fi
if [ "$(nnpz "$WD")" -lt 900 ]; then
  log "chờ GPU trống 50GB cho compute_weights"; wait_free 50000
  log "compute_weights (~4-6h)"
  run "w_${TAG}" train.compute_weights --model_name "$MODEL" --train_file "$TRAIN_FILE" \
    --phase1_dir "$P1" --weights_dir "$WD" --M 3 --disagreement_mode normalized \
    --cutoff_len 16384 --chunk_size 128 --preview 3 > "$LOG/compute_weights.log" 2>&1 \
    || { log "WEIGHTS FAIL"; exit 1; }
fi

# ---- 3b. V tại key step (đọc từ cache U/V, CPU) --------------------------------
VM="${VMASK:-${KD:-}}"
if [ -n "$VM" ] && [ ! -f "$BASE/v_keystep.json" ]; then
  CUDA_VISIBLE_DEVICES="" $PY scripts/v_at_keystep.py --weights_dir "$WD" --keystep_dir "$VM" \
    --out "$BASE/v_keystep.json" 2>&1 | tee "$LOG/v_keystep.log"
fi

# ---- 4. Phase 2 distill + eval student ------------------------------------------
DD=$BASE/distill
if [ ! -d "$DD/student" ]; then
  log "chờ GPU trống 76GB cho distill"; wait_free 76000
  log "distill (~2.5-4h)"
  run "d_${TAG}" train.phase2_distill --model_name "$MODEL" --train_file "$TRAIN_FILE" \
    --weights_dir "$WD" --teacher_adapter "$MD/merged" --lambda_kl 1.0 --cutoff_len 16384 \
    --out_dir "$DD" --device cuda > "$LOG/distill.log" 2>&1 || { log "DISTILL FAIL"; exit 1; }
fi
if [ ! -f "results/proposal_v2_abl_${TAG}_distill.json" ]; then
  log "chờ GPU trống 62GB cho eval"; wait_free 62000
  log "eval student"
  $PY -u scripts/titled_run.py "ev_${TAG}_dist" eval.run_eval --method "proposal_v2_abl_${TAG}_distill" \
    --adapter "$DD/student" --model_name "$MODEL" --engine vllm --gpu_memory_utilization 0.4 \
    > "$LOG/eval_student.log" 2>&1 || { log "EVAL STUDENT FAIL"; exit 1; }
fi
log "XONG -> results/proposal_v2_abl_${TAG}_{ties,distill}.json"
