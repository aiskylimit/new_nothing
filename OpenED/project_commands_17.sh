#!/usr/bin/env bash
# One of six queues (project_commands_13.sh ... 18.sh), each on its own GPU, h200-vllm trainer;
# project_commands_lib.sh holds the trainer settings, the job kinds and the knobs. GPU 6, in order:
#   1. Ours MAVEN perm3 and perm4
#   2. FewRel_sent perm4: Ours, the 7 distillation baselines, the 8 CL-LoRA methods
#   3. SDFT-CE perm4 on TACRED, RAMS, FewRel, MAVEN (project_commands_10.sh)
#   4. ACE ablations perm4 (project_commands_9.sh's configs not run yet)
#
#   bash project_commands_17.sh
#   DRY=1 bash project_commands_17.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p17
GPU=6
source ./project_commands_lib.sh

JOBS+=("ours|maven|3" "ours|maven|4")
sent_perm_jobs fewrel 4
for ds in tacred rams fewrel maven; do JOBS+=("sdftce|${ds}|4"); done
for c in ${ABL_CONFIGS}; do JOBS+=("abl|${c}|4"); done
run_all
