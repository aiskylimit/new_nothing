#!/usr/bin/env bash
# L_trans experiment driver (method B: next-step representation prediction), Qwen3-8B track.
# Config 4 of the experiment table: SFT + L_trans, lambda = 0.3. Same seed/LR/scheduler/LoRA as the
# SFT baseline (scripts/sft/sft_qwen3-8b.sh); the only change is the added objective.
# Paths are disjoint from the other drivers (data/qwen3-8b/train-*-trans.jsonl, checkpoints/trans-*,
# logs/trans-*, results/trans-*); only compare_results.py rewrites the shared summary files.
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) every phase script runs on (space-separated ids). Override: GPUS="0" ./project_commands_trans.sh
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0 1}"

# ============================ DATA ============================
# Phase 2 + 4 for a fresh server (only the raw downloads from download.txt present). Both are
# skipped when their output already exists; delete the file to force a rebuild.
#   data_qwen3-8b.sh     tokenizes + segments s1K-1.1 (CPU, ~1 min)      -> train-segmented.jsonl
#   vanilla_qwen3-8b.sh  all-ones NLL mask, no spectral capture needed   -> train-vanilla.jsonl
[[ -f data/qwen3-8b/train-segmented.jsonl ]] || bash scripts/data/data_qwen3-8b.sh
[[ -f data/qwen3-8b/train-vanilla.jsonl ]]   || bash scripts/masks/vanilla_qwen3-8b.sh
# Adds \n\n-step fields; the NLL mask is copied untouched. Seconds, CPU only.
bash scripts/trans/build_trans_qwen3-8b.sh vanilla
# bash scripts/trans/build_trans_qwen3-8b.sh spectral        # config 5 (SGL + L_trans): needs
#                                                            # capture_qwen3-8b.sh + masks_qwen3-8b.sh first

# ============================ TRAIN ============================
# Effective batch 32 regardless of GPU count (ga = 32 / n_gpu). Watch in logs/trans-*.log:
#   loss_trans / trans_cos      should fall / rise; trans_cos must clear trans_copy_cos
#   trans_raw_step_cos -> 1     step representations collapsing
#   trans_ztilde_norm -> 0      step-specific part vanishing
#   grad_trans_ratio            lambda*||dL_trans/dH|| / ||dL_NLL/dH|| at the transition layer; persistently > 0.3-0.5 => lower lambda
bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.3                 # config 4
# bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.1               # config 3
# bash scripts/trans/train_trans_qwen3-8b.sh spectral 0.3              # config 5 (best lambda)
# bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.3 shuffle       # config 6 (control)

# ============================ EVAL =============================
# eval_qwen3-8b.sh now defaults to a 4k context (max_model_len 4096, 3584 generated tokens) with
# thinking off; results/<tag>/ produced under the old 32k cap are NOT comparable -- re-eval the
# baselines below so the comparison table is apples to apples.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
TAG="trans-vanilla-l0.3-qwen3-8b"
# Adapter-only checkpoint like every other arm; trans_predictor.pt next to it is ignored by vLLM.
bash scripts/eval/eval_qwen3-8b.sh "checkpoints/${TAG}" "${TAG}"
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-qwen3-8b vanilla-qwen3-8b        # SFT baseline @4k
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-qwen3-8b iwc-stable-qwen3-8b  # SGL baseline @4k

# =========================== RESULTS ==========================
# Printed last so they are on screen when the driver exits (also in logs/eval-${TAG}.log and
# results/comparison-table.md): this arm's per-benchmark table, then the cross-arm comparison.
PY="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python"
PYTHONPATH="${BASE}/src" "${PY}" - "${TAG}" <<'PYEOF'
import json, sys
from pathlib import Path
from evaluate import print_summary_table
tag = sys.argv[1]
print_summary_table(tag, json.loads((Path("results") / tag / "summary.json").read_text()))
PYEOF
echo
"${PY}" -m sgl.eval.compare
