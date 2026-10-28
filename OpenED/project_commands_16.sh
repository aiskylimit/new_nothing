#!/usr/bin/env bash
# One of six queues (project_commands_13.sh ... 18.sh), each on its own GPU, h200-vllm trainer;
# project_commands_lib.sh holds the trainer settings, the job kinds and the knobs. GPU 5, in order:
#   1. Ours MAVEN perm1 (killed in project_commands_9.sh) and perm2
#   2. FewRel_sent perm3: Ours, the 7 distillation baselines, the 8 CL-LoRA methods
#   3. SDFT-CE perm3 on TACRED, RAMS, FewRel, MAVEN (project_commands_10.sh)
#   4. ACE ablations perm3 (project_commands_9.sh's configs not run yet)
#
#   bash project_commands_16.sh
#   DRY=1 bash project_commands_16.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p16
GPU=5
source ./project_commands_lib.sh

JOBS+=("ours|maven|1" "ours|maven|2")
sent_perm_jobs fewrel 3
for ds in tacred rams fewrel maven; do JOBS+=("sdftce|${ds}|3"); done
for c in ${ABL_CONFIGS}; do JOBS+=("abl|${c}|3"); done
run_all
