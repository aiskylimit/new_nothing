#!/usr/bin/env bash
# Qwen3-8B, LoRA, P-ALIGN data: IWC (entropy) and answer-gain arms -> train -> eval (3 sampling seeds).
# Thin entry point for project_commands_lora_palign.sh (see its header for the recipe and all env knobs).
# Same student, data (thinking off = empty <think></think>), LoRA (r16/alpha16) and 3 epochs / eff. batch 32 /
# lr 5e-5 as P-ALIGN/configs/qwen3_8b_palign_sft.yaml; eval is P-ALIGN's (4096 tok, n=3, T=0.6).
#   GPUS=0 bash project_commands_lora_palign_qwen3-8b.sh
#   ARMS="iwc gain" GPUS=0 bash project_commands_lora_palign_qwen3-8b.sh        # optional: add nll (P-ALIGN baseline)
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARMS="${ARMS:-iwc gain}"
exec bash "${BASE}/project_commands_lora_palign.sh" qwen3-8b
