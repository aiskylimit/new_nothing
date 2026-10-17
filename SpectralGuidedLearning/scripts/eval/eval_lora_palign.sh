#!/usr/bin/env bash
# Eval a LoRA adapter (or full checkpoint) with the P-ALIGN protocol (thinking OFF, AIME24/AIME25/
# AMC12/MATH500, n=3, T=0.6, top_p 0.9, rep. penalty 1.05, 4096 tokens -> Pass@1 / Pass@3). This is
# eval_r1-qwen-1.5b.sh with the base model given by BASE_MODEL: it is checkpoint-type agnostic (an
# adapter_config.json means a LoRA adapter loaded onto BASE_MODEL through vLLM) and model agnostic
# (Qwen2.5-7B-Instruct / Qwen3-8B render the same prompt as at train time, so --palign-prompt is a no-op there).
#   BASE_MODEL=Qwen/Qwen3-8B ./scripts/eval/eval_lora_palign.sh checkpoints/<arm>-qwen3-8b-palign <tag>
#   EVAL_SEED=43 RESULTS_DIR=results_evalseed BASE_MODEL=... ./scripts/eval/eval_lora_palign.sh <ckpt> <tag>
set -euo pipefail
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -n "${1:-}" && -n "${2:-}" ]] || { echo "usage: BASE_MODEL=<hf id> $0 <checkpoint> <tag>" >&2; exit 2; }
export BASE_MODEL="${BASE_MODEL:?set BASE_MODEL, e.g. Qwen/Qwen2.5-7B-Instruct}"
export ENABLE_THINKING=false
# A 7-8B bf16 model (~16GB) + KV cache fits easily in 0.5; 0.8 needs ~115GB free, which a shared GPU may not have.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
exec bash "${BASE_PATH}/scripts/eval/eval_r1-qwen-1.5b.sh" "$@"
