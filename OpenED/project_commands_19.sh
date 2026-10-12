#!/usr/bin/env bash
# The ACE tables still empty, perms 0-4, one run per GPU on GPUS [2 3 7]:
#   tab:ablation  pseudo-label filter: g2_nofilter ("Unfiltered") and g2_ground ("Grounding only"),
#                 no SD, about 1-2 h a run: they go first
#   tab:memory    Ours (f12_pl) with memory m = 0: job ours_b0 of project_commands_lib.sh, run
#                 ours_h2_f12_pl_perm<p>_ace_b0_v2_s42 (data ace_b0_perm<p>, KD on pseudo-labels only)
#   tab:sens_q    the PL confidence check keeping q = 30, 50, 100 of the candidates (configs q30 q50
#                 q100 of project_commands_lib.sh). q = 70 is the method's own setting: its row is
#                 ours_h12_sd_g1_full_perm<p>_ace_v2_s42, finished for all 5 perms, so not rerun here
#   tab:sens_sd   lambda_SD = 0.1, 0.3, 3.0 (configs g3_wsd01 g3_wsd03 g3_wsd30). lambda 0 and 1.0
#                 are a_nosd and g1_full, finished
#   tab:sens_span linear CKA instead of cosine (config g4_cka); cosine is g1_full, finished
#   tab:ablation  -span KD (config g4_nospan; the partial p9 runs of 09/10 are moved to _failed/)
# 55 runs. The SD runs (sens_span, -span KD, sens_q, sens_sd) take about 16 h each; they are handed
# out perm by perm, g4_cka and g4_nospan first, so every setting gets perm 0 first. All abl configs take ours_queue.sh's h12 flags, as the ACE ablations of
# project_commands_13.sh ... 17.sh, own task0. Those queues list the g2/g3 configs too: whichever
# queue reaches a run first trains it, and the other skips it (finished, or running elsewhere).
# One worker per GPU, each taking the next unclaimed run; logs/p19_gpu<g>_pool.log says what each
# one does. A run starts once its card has NEED_MB free, and takes torchrun ports 31500 + 10 x gpu,
# apart from p13-p18 (31000 + 10 x gpu).
#
# Trigger/argument F1: results/qwen3/ced/<run>/task4/log.txt.
# PL P / PL R (sens_q): gather_pl_logs.sh -> pl_logs/pl_quality_ace.log (tools/ced_pl_quality.py on
# the labels each run kept), plus the PL_QUALITY lines of each pl_task<t>.log. Both need the
# unstripped split data/ace_oracle_b10_perm<p> for every perm: built below when missing (HF_TOKEN
# from .env, the source dataset is private).
#
#   bash project_commands_19.sh
#   GPUS="7" bash project_commands_19.sh
#   DRY=1 bash project_commands_19.sh                 # print the plan, train nothing
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(${GPUS:-2 3 7})
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
for p in ${PERMS}; do   # no SD, short: first
    for c in g2_nofilter g2_ground; do JOBS19+=("abl|${c}|${p}"); done
    JOBS19+=("ours_b0|ace|${p}")
done
for p in ${PERMS}; do   # SD, about 16 h each: perm by perm, sens_span and -span KD first
    for c in g4_cka g4_nospan q30 q50 q100 g3_wsd01 g3_wsd03 g3_wsd30; do JOBS19+=("abl|${c}|${p}"); done
done

# a claim is a file bash creates with O_EXCL (see lock in the lib); cleared at each launch
CLAIMS=${LOCKS}/p19_claims
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
worker () {  # $1 = gpu
    local j n=0
    Q=p19_gpu$1; GPU=$1
    source ./project_commands_lib.sh
    PORT=$((31500 + 10 * GPU))
    log "=== ${Q}: ablation / sens_q / sens_sd / memory runs on gpu${GPU}, gen ${GEN_BACKEND} (DRY=${DRY}) ==="
    for j in "${JOBS19[@]}"; do
        ( set -o noclobber; : > "${CLAIMS}/${j//|/_}" ) 2>/dev/null || continue
        run_job "${j}" || n=$((n + 1))
    done
    log "=== ${Q} all done, ${n} failed: grep FAILED ${LOG} ==="
}
for g in "${GPUS[@]}"; do worker "${g}" & done
wait
