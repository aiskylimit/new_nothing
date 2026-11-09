#!/usr/bin/env bash
#
# Mot lenh duy nhat: dung moi truong, chay one_pass.py, in ket qua.
#
#   bash commands.sh
#
# Can san tren may (khong tai HuggingFace luc chay):
#   model  /mnt/local/_models/aiskylimit_new_nothingnew_2/Qwen2.5-7B-Instruct
#   train  /mnt/local/_data/aiskylimit_new_nothingnew_2/s1k/train.jsonl
#   bench  /mnt/local/_data/aiskylimit_new_nothingnew_2/<aime24|aime25|amc12|math500|gpqa>/test.jsonl
#
# Ghi de duong dan: MODEL_DIR=... TRAIN_JSONL=... BENCH_DIR=... bash commands.sh
# -----------------------------------------------------------------------------

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="${VENV:-/mnt/local/uvenvs/ssft_onepass}"
MODEL_DIR="${MODEL_DIR:-/mnt/local/_models/aiskylimit_new_nothingnew_2/Qwen2.5-7B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/local/_data/aiskylimit_new_nothingnew_2/s1k/train.jsonl}"
BENCH_DIR="${BENCH_DIR:-$ROOT/../palign/data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/rebuttal}"
EVAL_PYTHON="${EVAL_PYTHON:-/mnt/local/uvenvs/p-align/bin/python}"
RUN_ID="${RUN_ID:-ssft_qwen25_7b_seed42}"
GPU="${GPU:-0,1,2}"
STAMP="${VENV}/.setup_done_onepass_v1"

die() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }

[[ -d "$MODEL_DIR" ]] || die "Khong thay model local: $MODEL_DIR"
[[ -f "$TRAIN_JSONL" ]] || die "Khong thay train jsonl: $TRAIN_JSONL"
[[ -x "$EVAL_PYTHON" ]] || command -v "$EVAL_PYTHON" >/dev/null || die "Can EVAL_PYTHON tro toi env vLLM/math-verify"

# -----------------------------------------------------------------------------
# 1. Moi truong. one_pass train bang unsloth roi cham diem trong cung process,
#    nen dung env train. Eval bang vLLM trong subprocess EVAL_PYTHON rieng.
# -----------------------------------------------------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
  log "Tao venv: ${VENV}"
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -f "$STAMP" ]]; then
  log "Cai dependency train + cham diem"
  python -m pip install --upgrade pip
  python -m pip install -r "${ROOT}/requirements-sft.txt"
  python -m pip install "sympy==1.14.0" "regex==2024.11.6" "word2number==1.1"
  python -m pip install -e "${ROOT}/Eval/latex2sympy"
  python -c "import unsloth, torch, sympy, regex, word2number"
  touch "$STAMP"
else
  log "Moi truong da san: ${VENV}"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"
log "Train + eval (one_pass.py)"
echo "    model    : ${MODEL_DIR}"
echo "    train    : ${TRAIN_JSONL}"
echo "    bench    : ${BENCH_DIR}"
echo "    output   : ${OUTPUT_DIR}"
echo "    run_id   : ${RUN_ID}"
echo "    gpu      : ${GPU}"

python -u "${ROOT}/one_pass.py" \
  --model "$MODEL_DIR" \
  --train-data "$TRAIN_JSONL" \
  --data-dir "$BENCH_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --eval-backend vllm --eval-python "$EVAL_PYTHON" \
  --eval-n "${EVAL_N:-3}" --seed "${TRAIN_SEED:-42}" \
  --epochs "${EPOCHS:-3}" --lr "${LR:-5e-5}" \
  --nproc-per-node "${NPROC_PER_NODE:-3}" \
  --batch-size "${BATCH_SIZE:-11}" --grad-accum "${GRAD_ACCUM:-1}"

# -----------------------------------------------------------------------------
# 2. In ket qua. Checkpoint van nam tren dia neu eval dung giua chung.
# -----------------------------------------------------------------------------
SUMMARY="${OUTPUT_DIR}/${RUN_ID}/${RUN_ID}_summary.json"
COST="${OUTPUT_DIR}/${RUN_ID}/${RUN_ID}_cost.json"
[[ -f "$SUMMARY" ]] || die "Khong thay ${SUMMARY}"

log "Ket qua"
python - "$SUMMARY" "$COST" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
cost = json.load(open(sys.argv[2], encoding="utf-8")) if __import__("os").path.isfile(sys.argv[2]) else {}
print("  checkpoint : %s" % summary.get("checkpoint", ""))
if cost:
    print("  gpu hours  : %.3f" % cost.get("total_gpu_hours", 0.0))
    print("  retained   : %.1f%%" % cost.get("retained_token_pct", 0.0))
print("  %-12s %8s %8s %8s" % ("benchmark", "n", "pass@1", "pass@n"))
for name, row in (summary.get("benchmarks") or {}).items():
    print("  %-12s %8s %8s" % (name, row.get("pass@1", ""), row.get("pass@3", "")))
print()
print("  summary : %s" % sys.argv[1])
print("  cost    : %s" % sys.argv[2])
PY
