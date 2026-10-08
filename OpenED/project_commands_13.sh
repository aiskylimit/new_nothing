#!/usr/bin/env bash
# Sentence-level CRE on FewRel: project_commands_12.sh with DS=fewrel (every baseline plus Ours,
# 5 perms, gpus 0 1 5, 2 runs per GPU). See that file for the runs, data and knobs.
#
#   bash project_commands_13.sh
#   DRY=1 bash project_commands_13.sh                 # print the plan, train nothing
#
# Data: fewrel_all_new.tar.gz (datht/processed-new-cl-fewrel, 267,178,997 bytes), put next to this
# script by download.txt. project_commands_12.sh unpacks it to data/fewrel_sent_perm<p> and
# processed_data/fewrel_sent_perm<p> and ignores the old fewrel_all.tar.gz; it downloads nothing.
# Own claims, pool log (logs/p12_fewrel_pool.log) and ports, so it can run next to the TACRED one.
set -uo pipefail
DS=fewrel exec bash "$(dirname "$0")/project_commands_12.sh"
