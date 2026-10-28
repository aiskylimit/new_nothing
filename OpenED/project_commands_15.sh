#!/usr/bin/env bash
# One of six queues (project_commands_13.sh ... 18.sh), each on its own GPU, h200-vllm trainer;
# project_commands_lib.sh holds the trainer settings, the job kinds and the knobs. GPU 4, in order:
#   1. RAMS perm0 of the baselines that only have perms 1-4: RKL, DistiLLM (dist_queue.sh, shared
#      task0) and the 8 CL-LoRA methods (project_commands_11.sh's rams0); Ours MAVEN perm0
#   2. FewRel_sent perm2: Ours, the 7 distillation baselines, the 8 CL-LoRA methods
#   3. SDFT-CE perm2 on TACRED, RAMS, FewRel, MAVEN (project_commands_10.sh)
#   4. ACE ablations perm2 (project_commands_9.sh's configs not run yet)
#
#   bash project_commands_15.sh
#   DRY=1 bash project_commands_15.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p15
GPU=4
source ./project_commands_lib.sh

for m in rkl distillm; do JOBS+=("qdist|rams|${m}|0"); done
for m in ${SENT_CL}; do JOBS+=("cl|rams|${m}|0"); done
JOBS+=("ours|maven|0")
sent_perm_jobs fewrel 2
for ds in tacred rams fewrel maven; do JOBS+=("sdftce|${ds}|2"); done
for c in ${ABL_CONFIGS}; do JOBS+=("abl|${c}|2"); done
run_all
