#!/usr/bin/env bash
# One of six queues (project_commands_13.sh ... 18.sh), each on its own GPU, h200-vllm trainer;
# project_commands_lib.sh holds the trainer settings, the job kinds and the knobs. GPU 7, in order:
#   1. Ours FewRel (the old fewrel_perm data) perms 0-4 (perm0 killed in project_commands_11.sh)
#   2. Ours on GENEVA with Llama-3.2-1B and Gemma-3-1b, perms 0-4 (project_commands_11.sh's
#      backbones): these train in the new_nothingnew_2 checkout (NEW2) with its own, older trainer
#
#   bash project_commands_18.sh
#   DRY=1 bash project_commands_18.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p18
GPU=7
source ./project_commands_lib.sh

for p in 0 1 2 3 4; do JOBS+=("ours|fewrel|${p}"); done
for mt in llama gemma; do for p in 0 1 2 3 4; do JOBS+=("bb|${mt}|${p}"); done; done
run_all
