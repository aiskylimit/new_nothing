#!/usr/bin/env bash
# Re-run the GENEVA baselines that run.sh lost, on GPUs 4 and 6, alongside the running jobs.
#
#   bash project_commands_3.sh                 # GENEVA, perms 0-4
#   DRY=1 bash project_commands_3.sh           # print what it would do
#   PERMS="0 1" bash project_commands_3.sh     # only some perms
#
# What run.sh lost, per perm:
#   * dist_kd: OOM at task1 (micro batch 32, ~140 GB on a GPU shared with the ACE pool).
#     run.sh puts kd, sfkl, csd and amid in one sub-queue under set -e, so kd's crash also
#     skips sfkl, csd and amid: they never start.
#   * cllora_gainlora_o: OOM at task0 (micro batch 32, ~140 GB).
# Here every run uses micro batch 8 x accumulation 4 (effective batch 32, as before), and a
# failed run is logged and skipped instead of stopping the rest.
#
# run.sh is still going through perms 2-4 and will hit the same failures there. So each worker
# follows it: it waits until run.sh's own attempt at a perm has ended (run dir present, no live
# process), then takes over. A crashed partial run dir is moved to results/qwen3/ced/_failed/
# (not deleted): resuming is impossible because the micro batch changes.
#
# Two workers, one per GPU, each one run at a time:
#   GPU 4: dist kd, sfkl, csd, amid        GPU 6: cllora gainlora_o
# Distinct master ports: the ACE pool already uses 29500 + gpu.
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
DS=${DS:-geneva}
PERMS=${PERMS:-"0 1 2 3 4"}
GPU_DIST=${GPU_DIST:-4}
GPU_CL=${GPU_CL:-6}
SEED=${SEED:-42}
PROTOCOL=${PROTOCOL:-${DS}_v2}
POLL=${POLL:-300}
R=results/qwen3/ced
mkdir -p logs logs/_failed "${R}/_failed"
LOG=logs/${DS}_rerun.log
log () { echo "[rerun $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---------------------------------------------------------------- environment (as project_commands.sh)
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then set +u; source "${VENV}/bin/activate"; set -u; fi
PY=${PY:-$(command -v python || command -v python3)}
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
if [ -f models/Qwen3-0.6B/config.json ]; then
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

runsh_alive () { pgrep -f '(^|/| )run\.sh( |$)' > /dev/null; }
live () { pgrep -f -- "$1" > /dev/null; }   # $1 = regex on the command line
park () {  # $1 = run name: move a crashed partial run out of the way
    local dst="${R}/_failed/$1_$(date +%Y%m%d_%H%M)"
    log "  moving partial ${R}/$1 -> ${dst}"
    [ "${DRY}" = "1" ] || mv "${R}/$1" "${dst}"
}
wait_for () {  # $1 = description, $2 = ready-test function name, $3.. = its args. 1 = gave up
    local what=$1 test=$2; shift 2
    until "${test}" "$@"; do
        if ! runsh_alive; then
            "${test}" "$@" && return 0
            log "  give up on ${what}: run.sh is no longer running and never reached it"
            return 1
        fi
        [ "${DRY}" = "1" ] && { log "  DRY: ${what} not ready yet"; return 1; }
        sleep "${POLL}"
    done
}

# ---------------------------------------------------------------- distillation worker
dist_ready () {  # $1 = perm: shared task0 done and run.sh's own kd attempt has ended
    local kd="dist_kd_perm$1_${PROTOCOL}_s${SEED}"
    [ -f "${R}/dist_shared_task0_perm$1_${PROTOCOL}_s${SEED}/.complete" ] && [ -e "${R}/${kd}" ] \
        && ! live "run-name ${kd}( |$)"
}
run_dist () {  # $1 = method label $2 = kd-type $3 = perm, then optional runner flags
    local m=$1 kdt=$2 p=$3; shift 3
    local run="dist_${m}_perm${p}_${PROTOCOL}_s${SEED}"
    local pre="logs/${DS}_dist_${m}_perm${p}"
    [ -f "${R}/${run}/.complete" ] && { log "  skip ${run} (complete)"; return 0; }
    live "run-name ${run}( |$)" && { log "  skip ${run} (running elsewhere)"; return 0; }
    [ -e "${R}/${run}" ] && park "${run}"
    log "  start ${run} on gpu${GPU_DIST}"
    [ "${DRY}" = "1" ] && return 0
    local rc=0
    MASTER_PORT=$((29700 + GPU_DIST)) bash scripts/qwen/ced/run_ced_v2.sh \
        --run-name "${run}" --mode ce_kd --data-prefix "${DS}_b10_perm" --perm "${p}" \
        --kd-type "${kdt}" --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 8 --acc 4 \
        --greedy 1 --gpus "${GPU_DIST}" --start-task 1 \
        --task0-source-run "dist_shared_task0_perm${p}_${PROTOCOL}_s${SEED}" "$@" \
        > "${pre}_steps.log" 2>&1 || rc=$?
    : > "${pre}_results.log"   # same per-task log.txt dump as dist_queue.sh
    for f in $(find "${R}/${run}" -name log.txt 2>/dev/null | sort -V); do
        echo "===== ${f#${R}/${run}/} =====" >> "${pre}_results.log"; cat "${f}" >> "${pre}_results.log"
    done
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then
        log "  done  ${run}"
    else
        log "  FAILED ${run} (exit ${rc}), see ${pre}_steps.log"
    fi
}
dist_worker () {
    local p
    for p in ${PERMS}; do
        wait_for "dist perm${p}" dist_ready "${p}" || continue
        # kd complete without this script having run it means run.sh's sub-queue survived and
        # goes on to sfkl/csd/amid itself; starting them here too would collide with it
        if [ -f "${R}/dist_kd_perm${p}_${PROTOCOL}_s${SEED}/.complete" ] \
           && ! grep -q "start dist_kd_perm${p}_" "${LOG}"; then
            log "dist perm${p}: run.sh's own kd finished, its queue runs sfkl/csd/amid; skip"
            continue
        fi
        log "dist perm${p}: kd sfkl csd amid"
        run_dist kd kd "${p}"
        run_dist sfkl sfkl "${p}"
        run_dist csd csd "${p}"
        run_dist amid adaptive-amid "${p}" \
            --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5"
    done
    log "dist worker finished"
}

# ---------------------------------------------------------------- CL-LoRA worker
cl_ready () {  # $1 = perm: run.sh's own gainlora_o attempt has ended
    [ -e "${R}/cllora_gainlora_o_perm$1_${PROTOCOL}_s${SEED}" ] \
        && ! live "cl-method gainlora_o --data-root data/${DS}_b10_perm$1( |$)"
}
cl_worker () {
    local p run rc
    for p in ${PERMS}; do
        run="cllora_gainlora_o_perm${p}_${PROTOCOL}_s${SEED}"
        wait_for "${run}" cl_ready "${p}" || continue
        [ -f "${R}/${run}/.complete" ] && { log "  skip ${run} (complete)"; continue; }
        park "${run}"
        [ "${DRY}" = "1" ] || cp "logs/${DS}_cllora_gainlora_o_perm${p}_train.log" logs/_failed/ 2>/dev/null || true
        log "  start ${run} on gpu${GPU_CL}"
        [ "${DRY}" = "1" ] && continue
        rc=0
        bash scripts/qwen/ced/run_cllora.sh \
            --method gainlora_o --data-root "data/${DS}_b10_perm${p}" --num-tasks 5 \
            --rank 16 --alpha 64 --lr 2e-4 --epochs 5 \
            --batch-size 8 --grad-accum 4 --eval-batch-size 16 \
            --gpu "${GPU_CL}" --py "${PY}" --protocol "${PROTOCOL}" \
            >> "${LOG}" 2>&1 || rc=$?
        if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then
            log "  done  ${run}"
        else
            log "  FAILED ${run} (exit ${rc}), see logs/${DS}_cllora_gainlora_o_perm${p}_train.log"
        fi
    done
    log "cllora worker finished"
}

log "=== ${DS} perms '${PERMS}': dist on gpu${GPU_DIST}, gainlora_o on gpu${GPU_CL} (DRY=${DRY}) ==="
dist_worker & DPID=$!
cl_worker & CPID=$!
wait "${DPID}" "${CPID}"
log "=== all done; collect with: bash gather_logs.sh ${DS}_rerun_$(date +%Y%m%d_%H%M) ==="
