#!/usr/bin/env bash
# tab:sens_q: Ours on ACE perm0 with the PL confidence check keeping q = 30, 50, 100 and 70 (the
# method's own setting, rerun so its pseudo-label precision/recall reaches the log too) of the
# candidates. Same h12 flags and SD as the ACE ablations of project_commands_13.sh (abl jobs,
# configs q30 q50 q70 q100 in project_commands_lib.sh), own task0. One worker per GPU of GPUS
# [4 5 6], each taking the next unclaimed run; logs/p19_gpu<g>_pool.log says what each one does.
# GPUs 4-6 also carry p15-p17: a run starts once its card has NEED_MB free, and takes torchrun
# ports 31500 + 10 x gpu so it never meets theirs (31000 + 10 x gpu).
#
# Trigger/argument F1: results/qwen3/ced/ours_h12_sd_q<q>_perm0_ace_v2_s42/task4/log.txt.
# PL P / PL R: the PL_QUALITY lines of that run's pl_task<t>.log (tools/ced_pseudo_label.py), which
# need the unstripped split data/ace_oracle_b10_perm0: built below when missing (HF_TOKEN from
# .env, the source dataset is private). gather_logs.sh collects both files.
#
#   bash project_commands_19.sh
#   GPUS="4 5" bash project_commands_19.sh
#   DRY=1 bash project_commands_19.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(${GPUS:-4 5 6})
Q=p19
GPU=${GPUS[0]}
source ./project_commands_lib.sh

if [ ! -s data/ace_oracle_b10_perm0/streams.json ]; then
    log "building data/ace_oracle_b10_perm0 (no label stripping) for the PL_QUALITY lines"
    if [ "${DRY}" != "1" ]; then
        [ -f .env ] && [ -z "${HF_TOKEN:-}" ] && { set -a; . ./.env; set +a; }
        OPENED_BASE=$(pwd) HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "${PY}" tools/build_ced_perms.py \
            --cap 10 --oracle --perms 0 --out-prefix ace_oracle_b10_perm > "logs/${Q}_oracle.log" 2>&1 \
            || log "  oracle build FAILED (see logs/${Q}_oracle.log): runs go on, PL_QUALITY lines will say skipped"
    fi
fi

# a claim is a file bash creates with O_EXCL (see lock in the lib); cleared at each launch
CLAIMS=${LOCKS}/p19_claims
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
worker () {  # $1 = gpu
    local c n=0
    Q=p19_gpu$1; GPU=$1
    source ./project_commands_lib.sh
    PORT=$((31500 + 10 * GPU))
    log "=== ${Q}: tab:sens_q runs on gpu${GPU}, gen ${GEN_BACKEND} (DRY=${DRY}) ==="
    for c in q30 q50 q100 q70; do
        ( set -o noclobber; : > "${CLAIMS}/${c}" ) 2>/dev/null || continue
        run_job "abl|${c}|0" || n=$((n + 1))
    done
    log "=== ${Q} all done, ${n} failed: grep FAILED ${LOG} ==="
}
for g in "${GPUS[@]}"; do worker "${g}" & done
wait
