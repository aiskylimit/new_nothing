#!/usr/bin/env bash
# One of six queues (project_commands_13.sh ... 18.sh), each on its own GPU, h200-vllm trainer;
# project_commands_lib.sh holds the trainer settings, the job kinds and the knobs. GPU 0, in order:
#   1. reruns: TACRED_sent perm4 RKL CSD DistiLLM AMiD O-LoRA EPI TreeLoRA GainLoRA(InfLoRA)
#      (failed or killed in project_commands_12.sh); Ours TACRED perm3 (project_commands_11.sh)
#   2. FewRel_sent perm0: Ours, the 7 distillation baselines, the 8 CL-LoRA methods
#   3. SDFT-CE perm0 on TACRED, RAMS, FewRel, MAVEN (project_commands_10.sh)
#   4. ACE ablations perm0 (project_commands_9.sh's configs not run yet)
# FewRel_sent data: fewrel_all_new.tar.gz next to this script (download.txt), unpacked once by
# whichever queue gets there first; nothing is downloaded.
#
#   bash project_commands_13.sh
#   DRY=1 bash project_commands_13.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p13
GPU=0
source ./project_commands_lib.sh

for m in rkl csd distillm amid; do JOBS+=("sent_dist|tacred|${m}|4"); done
for m in olora epi tree gainlora_inf; do JOBS+=("sent_cl|tacred|${m}|4"); done
JOBS+=("ours|tacred|3")
sent_perm_jobs fewrel 0
for ds in tacred rams fewrel maven; do JOBS+=("sdftce|${ds}|0"); done
for c in ${ABL_CONFIGS}; do JOBS+=("abl|${c}|0"); done
run_all
