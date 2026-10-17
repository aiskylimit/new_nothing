#!/usr/bin/env bash
# One of six queues (project_commands_13.sh ... 18.sh), each on its own GPU, h200-vllm trainer;
# project_commands_lib.sh holds the trainer settings, the job kinds and the knobs. GPU 1, in order:
#   1. reruns: Ours TACRED_sent perm3 and perm4 (killed in project_commands_12.sh); GENEVA perm3
#      SRKL DistiLLM AMiD (OOM in project_commands_10.sh)
#   2. FewRel_sent perm1: Ours, the 7 distillation baselines, the 8 CL-LoRA methods
#   3. SDFT-CE perm1 on TACRED, RAMS, FewRel, MAVEN (project_commands_10.sh)
#   4. ACE ablations perm1 (project_commands_9.sh's configs not run yet)
#
#   bash project_commands_14.sh
#   DRY=1 bash project_commands_14.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p14
GPU=1
source ./project_commands_lib.sh

JOBS+=("sent_ours|tacred|3" "sent_ours|tacred|4")
for m in srkl distillm amid; do JOBS+=("qdist|geneva|${m}|3"); done
sent_perm_jobs fewrel 1
for ds in tacred rams fewrel maven; do JOBS+=("sdftce|${ds}|1"); done
for c in ${ABL_CONFIGS}; do JOBS+=("abl|${c}|1"); done
run_all
