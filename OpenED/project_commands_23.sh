#!/usr/bin/env bash
# Rebuttal runs, part 2 of 3 (project_commands_rebuttal_lib.sh), on GPUs 4 5, shortest first:
#   IncLoRA m=20, VPL+CE x1, O-LoRA m=5, TreeLoRA m=5, VPL+CE x5, CSD wall-clock, f12_pl m=5, CSD on Qwen3-4B
#   bash project_commands_23.sh
#   DRY=1 bash project_commands_23.sh                 # print the plan, train nothing
cd "$(dirname "$0")"
GPUS="4 5" PART=2 Q=p23 exec bash project_commands_rebuttal_lib.sh "$@"
