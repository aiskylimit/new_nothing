#!/usr/bin/env bash
# The last two tables no other queue covers, ACE, perms 0-4:
#   tab:memory  Ours (f12_pl) with memory m = 0: job ours_b0 of project_commands_lib.sh, run
#               ours_h2_f12_pl_perm<p>_ace_b0_v2_s42 (data ace_b0_perm<p>, KD on pseudo-labels only)
#   tab:sens_q  the rest of this header:
# Ours on ACE, perms 0-4, with the PL confidence check keeping q = 30, 50 and 100 of the
# candidates. Same h12 flags and SD as the ACE ablations of project_commands_13.sh (abl jobs,
# configs q30 q50 q100 in project_commands_lib.sh), own task0. q = 70 is the method's own setting:
# its row is ours_h12_sd_g1_full_perm<p>_ace_v2_s42 (project_commands.sh's "g1_full|1||", the same
# flags as config q70), finished for all 5 perms, so it is not rerun here. One worker per GPU of
# GPUS [0 1], each taking the next unclaimed run; logs/p19_gpu<g>_pool.log says what each one does.
# GPUs 0-1 also carry p13-p14: a run starts once its card has NEED_MB free, and takes torchrun
# ports 31500 + 10 x gpu so it never meets theirs (31000 + 10 x gpu).
#
# Trigger/argument F1: results/qwen3/ced/ours_h12_sd_q<q>_perm<p>_ace_v2_s42/task4/log.txt.
# PL P / PL R: gather_pl_logs.sh -> pl_logs/pl_quality_ace.log (tools/ced_pl_quality.py on the
# labels each run kept), plus the PL_QUALITY lines of each pl_task<t>.log. Both need the
# unstripped split data/ace_oracle_b10_perm<p> for every perm: built below when missing (HF_TOKEN
# from .env, the source dataset is private).
#
#   bash project_commands_19.sh
#   GPUS="0" bash project_commands_19.sh
#   DRY=1 bash project_commands_19.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(${GPUS:-7})
PERMS="0 1 2 3 4"
Q=p19
GPU=${GPUS[0]}
source ./project_commands_lib.sh

missing=""
for p in ${PERMS}; do [ -s "data/ace_oracle_b10_perm${p}/streams.json" ] || missing+=" ${p}"; done
if [ -n "${missing}" ]; then
    log "building data/ace_oracle_b10_perm{${missing# }} (no label stripping) for PL precision/recall"
    if [ "${DRY}" != "1" ]; then
        [ -f .env ] && [ -z "${HF_TOKEN:-}" ] && { set -a; . ./.env; set +a; }
        # shellcheck disable=SC2086   # one perm per word
        OPENED_BASE=$(pwd) HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "${PY}" tools/build_ced_perms.py \
            --cap 10 --oracle --perms ${missing} --out-prefix ace_oracle_b10_perm > "logs/${Q}_oracle.log" 2>&1 \
            || log "  oracle build FAILED (see logs/${Q}_oracle.log): runs go on, PL_QUALITY lines will say skipped"
    fi
fi

JOBS19=()
for p in ${PERMS}; do
    for c in q30 q50 q100; do JOBS19+=("abl|${c}|${p}"); done
    JOBS19+=("ours_b0|ace|${p}")
done

# a claim is a file bash creates with O_EXCL (see lock in the lib); cleared at each launch
CLAIMS=${LOCKS}/p19_claims
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
worker () {  # $1 = gpu
    local j n=0
    Q=p19_gpu$1; GPU=$1
    source ./project_commands_lib.sh
    PORT=$((31500 + 10 * GPU))
    log "=== ${Q}: tab:sens_q + tab:memory runs on gpu${GPU}, gen ${GEN_BACKEND} (DRY=${DRY}) ==="
    for j in "${JOBS19[@]}"; do
        ( set -o noclobber; : > "${CLAIMS}/${j//|/_}" ) 2>/dev/null || continue
        run_job "${j}" || n=$((n + 1))
    done
    log "=== ${Q} all done, ${n} failed: grep FAILED ${LOG} ==="
}
for g in "${GPUS[@]}"; do worker "${g}" & done
wait
