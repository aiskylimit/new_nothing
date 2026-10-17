#!/bin/bash
# Smoke test of the same env → data → train → merge → eval flow as
# project_commands.sh, using local Qwen2.5-0.5B-Instruct and tiny splits.
# Does not replace the 8B run. Production: bash project_commands.sh
#
#   bash project_commands_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/P-ALIGN}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
export PALIGN_ASSET_ROOT="$ASSET_ROOT"
export PALIGN_DATA_DIR="$DATA_DIR"

if [ -f "${PALIGN_VENV:-/mnt/local/uvenvs/p-align}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PALIGN_VENV:-/mnt/local/uvenvs/p-align}/bin/activate"
elif [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
else
  echo "no python env (PALIGN_VENV or /mnt/local/uvenvs/p-align or /venv/main)" >&2
  exit 1
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Make CUDA_VISIBLE_DEVICES indices match nvidia-smi (PCI order) instead of FASTEST_FIRST.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29331}"
EFFECTIVE_BATCH=2
PER_DEVICE_BS=1
SMOKE_N="${SMOKE_N:-1}"
SMOKE_MODEL="${SMOKE_MODEL:-$ROOT/models/Qwen2.5-0.5B-Instruct}"
SMOKE_LORA="${SMOKE_LORA:-output/smoke/palign-qwen2.5-0.5b-instruct-lora}"
SMOKE_MERGED="${SMOKE_MERGED:-output/smoke/palign-qwen2.5-0.5b-instruct-lora-merged}"
SMOKE_RAW="${SMOKE_RAW:-output/smoke/raw}"

if [ ! -f "$SMOKE_MODEL/config.json" ]; then
  echo "missing $SMOKE_MODEL (download Qwen/Qwen2.5-0.5B-Instruct first)" >&2
  exit 1
fi

python -c "import torch, transformers, llamafactory, vllm; print('env ok')"

mkdir -p "$DATA_DIR/raw" output/log output/result "$SMOKE_RAW" output/smoke
if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ] && [ -f "$DATA_DIR/palign_sft_qwen2.5-7b.json.gz" ]; then
  gunzip -kc "$DATA_DIR/palign_sft_qwen2.5-7b.json.gz" > "$DATA_DIR/palign_sft_qwen2.5-7b.json"
fi
if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ]; then
  echo "missing local train file $DATA_DIR/palign_sft_qwen2.5-7b.json" >&2
  exit 1
fi
for f in "$DATA_DIR/raw/aime25.jsonl" "$DATA_DIR/raw/aime24.jsonl" "$DATA_DIR/raw/amc12.jsonl" "$DATA_DIR/raw/math500.jsonl"; do
  if [ ! -f "$f" ]; then
    python src/fetch_eval.py
    break
  fi
done

python - <<PY
from pathlib import Path
n = int("${SMOKE_N}")
src = Path("${DATA_DIR}") / "raw"
dst = Path("${SMOKE_RAW}")
for name in ("aime25", "aime24", "amc12", "math500"):
    lines = [ln for ln in (src / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()][:n]
    (dst / f"{name}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"smoke slice {name}: {len(lines)}")
PY

WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GRAD_ACCUM=$((EFFECTIVE_BATCH / (PER_DEVICE_BS * WORLD_SIZE)))
echo "SMOKE SFT nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM} model=${SMOKE_MODEL}"
# --standalone: cluster pods export PET_RDZV_* which otherwise hangs single-node rendezvous
if [ "$NNODES" -eq 1 ]; then
  LAUNCH_ARGS=(--standalone --nproc_per_node "$NPROC_PER_NODE")
else
  LAUNCH_ARGS=(--nproc_per_node "$NPROC_PER_NODE" --nnodes "$NNODES" --node_rank "$RANK"
               --rdzv_backend static --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT")
fi
torchrun "${LAUNCH_ARGS[@]}" \
  src/train.py configs/qwen3_8b_palign_sft.yaml \
  model_name_or_path="$SMOKE_MODEL" \
  template=qwen \
  output_dir="$SMOKE_LORA" \
  cutoff_len=1024 \
  max_samples=4 \
  max_steps=2 \
  preprocessing_num_workers=2 \
  dataloader_num_workers=0 \
  save_strategy=steps \
  save_steps=2 \
  save_total_limit=1 \
  gradient_checkpointing=false \
  gradient_accumulation_steps="$GRAD_ACCUM"

llamafactory-cli export configs/qwen3_8b_palign_export.yaml \
  model_name_or_path="$SMOKE_MODEL" \
  template=qwen \
  adapter_name_or_path="$SMOKE_LORA" \
  export_dir="$SMOKE_MERGED" \
  export_size=2

python src/test.py \
  --model "$SMOKE_MERGED" \
  --input_files "$SMOKE_RAW/aime25.jsonl" "$SMOKE_RAW/aime24.jsonl" "$SMOKE_RAW/amc12.jsonl" "$SMOKE_RAW/math500.jsonl" \
  --output_files output/result/aime25.jsonl output/result/aime24.jsonl output/result/amc12.jsonl output/result/math500.jsonl \
  --batch_size 8 \
  --n 1 \
  --temperature 0.6 \
  --top_p 0.9 \
  --repetition_penalty 1.05 \
  --max_tokens 1024

for f in output/result/aime25.jsonl output/result/aime24.jsonl output/result/amc12.jsonl output/result/math500.jsonl; do
  python src/evaluation.py --input_path "$f" --output_path "${f%.jsonl}_scored.jsonl"
done
python src/report.py --out output/eval_results_smoke.txt
echo "wrote output/eval_results_smoke.txt"
