#!/usr/bin/env bash
# IWC experiment driver -- long-CoT data -> spectral capture -> IWC masks -> IWC SFT -> EVAL -> compare.
# Runs alongside project_commands.sh (answer-only arm) on a DIFFERENT GPU: every path below is
# disjoint from that driver (data/<track>/ vs data/<track>-answer/, checkpoints/iwc-*, logs/iwc-*,
# results/iwc-*). Only compare_results.py rewrites the shared results/comparison-table.md and
# results/eval-summary.json, and it regenerates them from every results/<tag>/ each time, so
# whichever driver finishes last simply produces the fuller table.
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Which GPU(s) every phase script runs on (space-separated ids). Override: GPUS="0" ./project_commands2.sh
# Must NOT be the GPU project_commands.sh is training on: eval reserves 90% of its GPU for vLLM.
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

# ============================ TRAIN ============================
# per track: data -> capture (spectral + entropy) -> IWC weights/masks -> IWC SFT (DeepSpeed/HF path, 1 GPU).
# IWC and IWC-Stable share exactly the spectral-selected token set; only weights differ.
# Each script activates its own env (main env spectral_guided_learning for every step here),
# so start this driver with no venv active.
# qwen25-7b: data + capture + IWC masks already on disk from the earlier run -- retrain only if
# the new train_iwc_unsloth.sh hyperparameters (ga 32, lora r 16) should apply to this track too.
# bash scripts/data/data_qwen25-7b.sh
# bash scripts/capture/capture_qwen25-7b.sh
# bash scripts/masks/iwc_qwen25-7b.sh
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc-stable

# qwen3-8b: data + capture (spectral-strengths.parquet with step_entropies) already on disk.
# Both steps are idempotent (fixed seed / npz resume) but each reloads the model -- keep them off
# unless data/qwen3-8b/ was wiped.
# bash scripts/data/data_qwen3-8b.sh
# bash scripts/capture/capture_qwen3-8b.sh

# IWC-Stable weights (Eq. 13-16), deliberately gentle: lambda 0.5 halves the tilt toward
# high-entropy steps, tau 2.0 flattens the softmax, clip 1.0 bounds any step to ~e^1 of the mean.
# Token mass is preserved per sample (weight_mass_ratio must print 1.000000), so the only thing
# that differs from the matched spectral arm (checkpoints/iwc-unsloth-qwen3-8b, r16/alpha16/ga32,
# whose weights were never applied -- see train_sft_unsloth.py) is the within-sample allocation.
# Rebuilds data/qwen3-8b/train-iwc.jsonl and train-iwc-stable.jsonl; leaves train-spectral untouched.
export IWC_INTERPOLATION="${IWC_INTERPOLATION:-0.5}"
export IWC_TEMPERATURE="${IWC_TEMPERATURE:-2.0}"
export IWC_CLIP="${IWC_CLIP:-1.0}"
bash scripts/masks/iwc_qwen3-8b.sh

# Train through the DeepSpeed/HF path ONLY: train_sft.py's MaskedSFTTrainer applies loss_weights
# via masked_cross_entropy. train_sft_unsloth.py uses the stock Trainer + Unsloth fused loss,
# which silently drops loss_weights -- its "iwc" checkpoints are plain spectral SFT.
# GPUS=<one id> gives bs1 x ga32 = effective batch 32, LoRA r16/alpha16 -- matched to the
# vanilla-unsloth arm. Materializes fp32 logits (T x V): fine on B200; on 80GB drop --max-seq-len to 24576.
bash scripts/iwc/train_iwc.sh qwen3-8b iwc-stable
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc            # Eq. 11 baseline: not mass-preserving, skip as main arm
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc    # BROKEN for IWC (weights ignored) -- do not use
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc-stable

# ============================ EVAL =============================
# Leave the unsloth train env so each eval script activates the vLLM env (spectral-guided-learning).
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-unsloth-qwen25-7b iwc-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-unsloth-qwen25-7b iwc-stable-unsloth-qwen25-7b

bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-qwen3-8b iwc-stable-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-qwen3-8b iwc-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-unsloth-qwen3-8b iwc-unsloth-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-unsloth-qwen3-8b iwc-stable-unsloth-qwen3-8b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json (regenerated from all results/<tag>/)
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" -m sgl.eval.compare
