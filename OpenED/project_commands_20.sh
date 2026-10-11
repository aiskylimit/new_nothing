#!/usr/bin/env bash
# The 10 runs of project_commands_18.sh that failed on 10/10 ("no Llama/Gemma checkout at NEW2"):
# Ours (f12_pl) on GENEVA with Llama-3.2-1B-Instruct and Gemma-3-1b-it, perms 0-4
# (tab:backbones_geneva). They now train in this checkout (chat_format.py, MODEL_TYPE in
# run_ced_v2.sh), job bb of project_commands_lib.sh: span layers 13 14 16 (Llama) / 20 23 26
# (Gemma), micro-batch 8 x 4 as their baselines, HF generation. GPU 7, one run at a time.
# Needs models/Llama-3.2-1B-Instruct and models/gemma-3-1b-it (download.txt) and data/geneva_b10_perm*;
# the tokenised GENEVA splits come from geneva_all_<llama|gemma>.tar.gz (download.txt, HF
# datht/<llama|gemma>-processed-cl-geneva), unpacked here; without it they are tokenised on first use.
#
#   bash project_commands_20.sh
#   DRY=1 bash project_commands_20.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
Q=p20
GPU=7
source ./project_commands_lib.sh

for mt in llama gemma; do   # processed_data/geneva_b10_perm<p>/<t>/<mt>, same rows as tokenising here
    tgz=geneva_all_${mt}.tar.gz
    if [ -e "${tgz}" ] && [ ! -f "processed_data/geneva_b10_perm4/4/${mt}/train_0.idx" ]; then
        log "unpacking ${tgz}"
        [ "${DRY}" = "1" ] || tar xzf "${tgz}" processed_data || log "  not unpacked: tokenising on first use instead"
    fi
done

for mt in llama gemma; do for p in 0 1 2 3 4; do JOBS+=("bb|${mt}|${p}"); done; done
run_all
