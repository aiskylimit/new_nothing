#!/usr/bin/env bash
# Rebuttal runs: rerun of TreeLoRA on Qwen3-4B (perms 0-4), the 5 jobs of project_commands_24.sh
# that ran out of GPU memory at 32 rows per step. project_commands_rebuttal_lib.sh now runs them
# as 8 x 4 (32 per update). Same GPUs 6 7 as p24; the failed run folders are moved to _failed.
#   bash project_commands_25.sh
#   DRY=1 bash project_commands_25.sh                 # print the plan, train nothing
cd "$(dirname "$0")"
GPUS="6 7" PART=3 STEPS=q4b Q=p25 exec bash project_commands_rebuttal_lib.sh "$@"
