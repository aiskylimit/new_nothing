#!/usr/bin/env bash
# Sentence-level CRE on FewRel: project_commands_12.sh with DS=fewrel (every baseline plus Ours,
# 5 perms, gpu 5, 2 runs at a time, h200-vllm trainer with PHYS_BS 32). See that file for the
# runs, data and knobs.
#
#   bash project_commands_13.sh
#   DRY=1 bash project_commands_13.sh                 # print the plan, train nothing
#
# Data: fewrel_all_new.tar.gz (datht/processed-new-cl-fewrel, 267,178,997 bytes), put next to this
# script by download.txt. project_commands_12.sh unpacks it to data/fewrel_sent_perm<p> and
# processed_data/fewrel_sent_perm<p> and ignores the old fewrel_all.tar.gz; it downloads nothing.
# Own claims, pool log (logs/p12_fewrel_pool.log) and ports, so it can run next to the TACRED one
# (gpu 4); GPUS="..." still overrides gpu 5.
set -uo pipefail
DS=fewrel GPUS=${GPUS:-5} exec bash "$(dirname "$0")/project_commands_12.sh"
