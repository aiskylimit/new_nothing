#!/usr/bin/env bash
# Re-run what failed or was skipped on 05/10 (logs of 15:21 UTC), one job at a time per GPU.
#
#   bash project_commands_4.sh                   # GPUs 5 and 7
#   DRY=1 bash project_commands_4.sh             # print the plan, train nothing
#   GPU_A=5 GPU_B=7 bash project_commands_4.sh
#
# The failures:
#   * ACE ours_h12_sd_c_nopl perm2: died in task3 (project_commands.sh pool, gpu6).
#   * GENEVA cllora inclora perm4: OOM (run.sh, gpu6); tree, epi, gainlora_o perm4 never ran
#     because they sat behind it in the same run.sh sub-queue.
#   * GENEVA dist shared task0 perm3 and perm4: OOM (run.sh, gpu4), so none of the seven
#     distillation baselines ran for those perms.
# All three OOMs came from stacking: the ACE pool (~34 GB per GPU) + run.sh + a
# project_commands_3.sh rerun (58-69 GB) on the same GPU. Hence GPUs 5 and 7 by default:
# project_commands_3.sh still runs on gpu4 and a TACRED CL-LoRA queue on gpu6, so 5 and 7
# only carry their ACE pool job. One job at a time per GPU here, so at most ACE + one run.
#
# Batch sizes match what the finished perms of the same method used, effective batch 32:
#   shared task0 32x1, rkl/srkl/distillm 16x2 (dist_queue.sh), kd/sfkl/csd/amid 8x4
#   (project_commands_3.sh), inclora/tree/epi 32x1 (run_all_cllora.sh), gainlora_o 8x4.
#
# A crashed partial run dir is moved to results/qwen3/ced/_failed/, never deleted. The one
# exception is c_nopl perm2, which resumes from its last finished task (task2): same batch,
# so the manifest allows it, and it saves the ~8 h of tasks 1-2.
#
#   GPU_A: c_nopl perm2 (resume), then GENEVA dist perm3
#   GPU_B: GENEVA cllora perm4 (inclora tree epi gainlora_o), then GENEVA dist perm4
# Safe to re-run: finished runs are skipped.
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
GPU_A=${GPU_A:-5}
GPU_B=${GPU_B:-7}
SEED=${SEED:-42}
R=results/qwen3/ced
mkdir -p logs "${R}/_failed"
LOG=logs/rerun4.log
log () { echo "[rerun4 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

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

live () { pgrep -f -- "$1" > /dev/null; }   # $1 = regex on the command line
park () {  # $1 = run name: move a crashed partial run out of the way
    local dst="${R}/_failed/$1_$(date +%Y%m%d_%H%M)"
    log "  moving partial ${R}/$1 -> ${dst}"
    [ "${DRY}" = "1" ] || mv "${R}/$1" "${dst}"
}
ready () {  # $1 = run name, $2 = regex of its live process. 0 = should be trained here
    [ -f "${R}/$1/.complete" ] && { log "  skip $1 (complete)"; return 1; }
    live "$2" && { log "  skip $1 (running elsewhere)"; return 1; }
    return 0
}
outcome () {  # $1 = run name $2 = exit code $3 = log to point at
    if [ "$2" -eq 0 ] && [ -f "${R}/$1/.complete" ]; then log "  done  $1"; return 0; fi
    log "  FAILED $1 (exit $2), see $3"; return 1
}

# ---------------------------------------------------------------- ACE c_nopl perm2 (resume)
ace_c_nopl_perm2 () {  # $1 = gpu
    local run="ours_h12_sd_c_nopl_perm2_ace_v2_s42" rc=0
    ready "${run}" "run-name ${run}( |$)" || return 0
    log "  resume ${run} on gpu$1 (from its last finished task)"
    [ "${DRY}" = "1" ] && return 0
    # what project_commands.sh launches for c_nopl, with RESUME=1
    PERM=2 GPU="$1" DATA_PREFIX=ace_b10_perm SEED="${SEED}" PROTOCOL=ace_v2 \
    OURS_VARIANT=h12 OURS_SD=1 SD_ARGS="" RUN_SUFFIX=_c_nopl RESUME=1 \
    MASTER_PORT=$((29800 + $1)) \
        bash scripts/qwen/ced/ours_queue.sh --pl 0 >> "${LOG}" 2>&1 || rc=$?
    outcome "${run}" "${rc}" "logs_${run}.log"
}

# ---------------------------------------------------------------- GENEVA CL-LoRA perm4
cllora () {  # $1 = gpu $2 = method $3 = perm $4 = micro batch $5 = grad accumulation
    local run="cllora_$2_perm$3_geneva_v2_s${SEED}" rc=0
    ready "${run}" "cl-method $2 --data-root data/geneva_b10_perm$3( |$)" || return 0
    [ -e "${R}/${run}" ] && park "${run}"
    log "  start ${run} on gpu$1 (batch $4x$5)"
    [ "${DRY}" = "1" ] && return 0
    bash scripts/qwen/ced/run_cllora.sh \
        --method "$2" --data-root "data/geneva_b10_perm$3" --num-tasks 5 \
        --rank 16 --alpha 64 --lr 2e-4 --epochs 5 \
        --batch-size "$4" --grad-accum "$5" --eval-batch-size 16 \
        --gpu "$1" --py "${PY}" --protocol geneva_v2 --seed "${SEED}" \
        >> "${LOG}" 2>&1 || rc=$?
    outcome "${run}" "${rc}" "logs/geneva_cllora_$2_perm$3_train.log"
}

# ---------------------------------------------------------------- GENEVA distillation, one perm
dist_queue_one () {  # $1 = gpu $2 = perm $3 = method: dist_queue.sh's own batch sizes
    local run="dist_$3_perm$2_geneva_v2_s${SEED}" rc=0
    ready "${run}" "run-name ${run}( |$)" || return 0
    [ -e "${R}/${run}" ] && park "${run}"
    log "  start ${run} on gpu$1"
    [ "${DRY}" = "1" ] && return 0
    PERM="$2" GPU="$1" PROTOCOL=geneva_v2 SEED="${SEED}" DATA_PREFIX=geneva_b10_perm DIST_METHODS="$3" \
    MASTER_PORT=$((29800 + $1)) \
        bash scripts/qwen/ced/dist_queue.sh >> "${LOG}" 2>&1 || rc=$?
    outcome "${run}" "${rc}" "logs/geneva_dist_$3_perm$2_steps.log"
}
dist_small () {  # $1 = gpu $2 = perm $3 = method label $4 = kd-type, then runner flags: 8x4
    local g=$1 p=$2 m=$3 kdt=$4; shift 4
    local run="dist_${m}_perm${p}_geneva_v2_s${SEED}" pre="logs/geneva_dist_${m}_perm${p}" rc=0
    ready "${run}" "run-name ${run}( |$)" || return 0
    [ -e "${R}/${run}" ] && park "${run}"
    log "  start ${run} on gpu${g} (8x4)"
    [ "${DRY}" = "1" ] && return 0
    MASTER_PORT=$((29800 + g)) bash scripts/qwen/ced/run_ced_v2.sh \
        --run-name "${run}" --mode ce_kd --data-prefix geneva_b10_perm --perm "${p}" \
        --kd-type "${kdt}" --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 8 --acc 4 \
        --greedy 1 --gpus "${g}" --start-task 1 \
        --task0-source-run "dist_shared_task0_perm${p}_geneva_v2_s${SEED}" "$@" \
        > "${pre}_steps.log" 2>&1 || rc=$?
    : > "${pre}_results.log"   # same per-task log.txt dump as dist_queue.sh
    for f in $(find "${R}/${run}" -name log.txt 2>/dev/null | sort -V); do
        echo "===== ${f#${R}/${run}/} =====" >> "${pre}_results.log"; cat "${f}" >> "${pre}_results.log"
    done
    outcome "${run}" "${rc}" "${pre}_steps.log"
}
geneva_dist_perm () {  # $1 = gpu $2 = perm
    local g=$1 p=$2 shared="dist_shared_task0_perm$2_geneva_v2_s${SEED}"
    log "GENEVA dist perm${p} on gpu${g}"
    # dist_queue.sh refuses an incomplete shared task0, and these two are crashed ones
    if [ -e "${R}/${shared}" ] && [ ! -f "${R}/${shared}/.complete" ]; then
        live "run-name ${shared}( |$)" && { log "  ${shared} is training elsewhere; skip perm${p}"; return; }
        park "${shared}"
    fi
    # rkl first: dist_queue.sh trains the shared task0 before it
    dist_queue_one "${g}" "${p}" rkl
    if [ "${DRY}" != "1" ] && [ ! -f "${R}/${shared}/.complete" ]; then
        log "  shared task0 perm${p} did not finish; skip the rest of perm${p}"; return
    fi
    dist_queue_one "${g}" "${p}" srkl
    dist_queue_one "${g}" "${p}" distillm
    dist_small "${g}" "${p}" kd kd
    dist_small "${g}" "${p}" sfkl sfkl
    dist_small "${g}" "${p}" csd csd
    dist_small "${g}" "${p}" amid adaptive-amid \
        --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5"
}

# ---------------------------------------------------------------- two workers
worker_a () {
    ace_c_nopl_perm2 "${GPU_A}"
    geneva_dist_perm "${GPU_A}" 3
    log "worker A (gpu${GPU_A}) finished"
}
worker_b () {
    log "GENEVA cllora perm4 on gpu${GPU_B}"
    cllora "${GPU_B}" inclora 4 32 1
    cllora "${GPU_B}" tree 4 32 1
    cllora "${GPU_B}" epi 4 32 1
    cllora "${GPU_B}" gainlora_o 4 8 4
    geneva_dist_perm "${GPU_B}" 4
    log "worker B (gpu${GPU_B}) finished"
}

log "=== rerun4: worker A on gpu${GPU_A}, worker B on gpu${GPU_B} (DRY=${DRY}) ==="
worker_a & APID=$!
worker_b & BPID=$!
wait "${APID}" "${BPID}"
log "=== all done; FAILED lines above need a look: grep FAILED ${LOG} ==="
