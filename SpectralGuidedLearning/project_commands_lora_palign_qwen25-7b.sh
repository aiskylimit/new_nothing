#!/usr/bin/env bash
# Qwen2.5-7B-Instruct, LoRA, P-ALIGN data: IWC (entropy) and answer-gain arms -> train -> eval (3 sampling seeds).
# Thin entry point for project_commands_lora_palign.sh (see its header for the recipe and all env knobs).
#   GPUS=0 bash project_commands_lora_palign_qwen25-7b.sh
#   ARMS="iwc gain nll" GPUS=0 bash project_commands_lora_palign_qwen25-7b.sh   # + P-ALIGN NLL baseline
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARMS="${ARMS:-iwc gain}"
exec bash "${BASE}/project_commands_lora_palign.sh" qwen25-7b
