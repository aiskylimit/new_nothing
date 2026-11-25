#!/usr/bin/env bash
# Rebuttal runs, part 3 of 3 (project_commands_rebuttal_lib.sh), on GPUs 6 7, shortest first:
#   CE x1, CE x5, oracle + CE, O-LoRA m=20, TreeLoRA m=20, oracle + f12_pl, TreeLoRA on Qwen3-4B
#   bash project_commands_24.sh
#   DRY=1 bash project_commands_24.sh                 # print the plan, train nothing
cd "$(dirname "$0")"
GPUS="6 7" PART=3 Q=p24 exec bash project_commands_rebuttal_lib.sh "$@"
