#!/usr/bin/env bash
# Rebuttal runs, part 1 of 3 (project_commands_rebuttal_lib.sh), on GPUs 0 1, shortest first:
#   joint, IncLoRA m=5, f12_pl, f12_pl m=20, f12_pl on Qwen3-4B
#   bash project_commands_22.sh
#   DRY=1 bash project_commands_22.sh                 # print the plan, train nothing
cd "$(dirname "$0")"
GPUS="0 1" PART=1 Q=p22 exec bash project_commands_rebuttal_lib.sh "$@"
