#!/usr/bin/env bash
# Rebuttal runs: Ours (f12_pl) on Qwen3-4B, perms 4 3 2 1, the rest of project_commands_22.sh's
# 4B step (its joint, IncLoRA m=5, f12_pl, f12_pl m=20 and 4B perm 0 ran on another server).
# PART=1 keeps only f12_4b of the q4b step (CSD-4B is in 23, TreeLoRA-4B in 24/25).
# GPUs 6 7 unless GPUS is set.
#   bash project_commands_26.sh
#   GPUS="0 1" bash project_commands_26.sh
#   DRY=1 bash project_commands_26.sh                 # print the plan, train nothing
cd "$(dirname "$0")"
GPUS="${GPUS:-6 7}" PART=1 STEPS=q4b PERMS="4 3 2 1" Q=p26 exec bash project_commands_rebuttal_lib.sh "$@"
