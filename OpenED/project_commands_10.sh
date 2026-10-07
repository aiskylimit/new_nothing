#!/usr/bin/env bash
# Re-run what failed on this host up to 07/10 14:57 UTC (collected_logs all3). Every failure in
# those logs was a CUDA OOM: jobs from several queues sharing a GPU, and the SDFT + CE pool of
# project_commands_6.sh at micro-batch 32 peaking at 145 GB of a 178 GB card.
#
#   bash project_commands_10.sh                      # gpus 4 and 5, one job per GPU
#   DRY=1 bash project_commands_10.sh                # print the plan, train nothing
#   GPUS="4 5 6" STEPS="cllora_ours ace" bash project_commands_10.sh
#
# Stop project_commands_6.sh (SDFT + CE, on gpus 4 and 5) before starting this: it keeps going at
# micro-batch 32. Its runs that are still training then die and are retrained here at 8 x 4.
#
# Steps (STEPS, in this order; default all but sdft):
#   cllora_ours  IncLoRA and TreeLoRA + Ours, perms 3 and 4 (project_commands_5.sh, OOM 06/10)
#   ace          ACE m_omask perm4 and d_full perms 1-4 (project_commands.sh pool, OOM 07/10)
#   geneva3      the seven GENEVA distillation baselines on perm3, never run because its shared
#                task0 kept failing until 07/10; batch sizes as the other perms (project_commands_4.sh)
#   sdftce       every SDFT + CE run (project_commands_6.sh, dist_sdftce_*) that is not finished
#   sdft         pure SDFT (dist_sdft_*), not in the default: replaced by sdftce on 07/10 because it
#                collapses (ACE Last 21.1, GENEVA 3.7, MAVEN 0)
#
# Each run trains its own task0, so nothing reads a checkpoint an earlier run left on disk (they
# can be deleted at any time). The one exception is geneva3: the distillation baselines share one
# task0 per perm by design, and dist_queue.sh trains it when it is missing or has lost its model.
# Flags are those of the queue each run comes from, so a rerun matches its finished siblings, with
# one change: SDFT runs use micro-batch 8 x 4 (SD_MB) instead of 32 x 1, same effective batch 32.
# Run names are unchanged, so the original queues skip what finishes here.
# A finished run is skipped, a run whose name is on a live process's command line is left alone,
# and a crashed partial one is moved to results/qwen3/ced/_failed/ (never deleted) and retrained.
#
# Knobs: GPUS ["4 5"], SLOTS jobs per GPU [1], NEED_MB free MiB a job waits for [60000],
# STEPS ["cllora_ours ace geneva3 sdftce"], SD_MB SDFT micro-batch [8], SEED [42].
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
GPUS=(${GPUS:-4 5})
SLOTS=${SLOTS:-1}
NEED_MB=${NEED_MB:-60000}
STEPS=${STEPS:-"cllora_ours ace geneva3 sdftce"}
SD_MB=${SD_MB:-8}
SEED=${SEED:-42}
R=results/qwen3/ced
CLAIMS=logs/p10_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/p10_pool.log
log () { echo "[p10 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---------------------------------------------------------------- environment (as project_commands.sh)
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then set +u; source "${VENV}/bin/activate"; set -u; fi
PY=${PY:-$(command -v python || command -v python3)}
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
if [ -f models/Qwen3-0.6B/config.json ]; then
    MODEL_PATH=models/Qwen3-0.6B
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

# ---------------------------------------------------------------- jobs: kind|name|dataset|perm
JOBS=()
want () { [[ " ${STEPS} " == *" $1 "* ]]; }
if want cllora_ours; then
    for m in inclora tree; do for p in 3 4; do JOBS+=("cllora_ours|${m}|ace|${p}"); done; done
fi
if want ace; then
    JOBS+=("ace|m_omask|ace|4")
    for p in 1 2 3 4; do JOBS+=("ace|d_full|ace|${p}"); done
fi
want geneva3 && JOBS+=("geneva3|dist|geneva|3")
for kind in sdftce sdft; do
    want "${kind}" || continue
    for ds in ace geneva tacred rams fewrel maven; do   # small -> large, as project_commands_6.sh
        for p in 0 1 2 3 4; do JOBS+=("${kind}|${kind}|${ds}|${p}"); done
    done
done

run_of () {  # $1 kind $2 name $3 dataset $4 perm -> run name, as the original queue names it
    case $1 in
        cllora_ours) echo "cllora_$2_ours_perm$4_ace_v2_s${SEED}" ;;
        ace)         echo "ours_h12_sd_$2_perm$4_ace_v2_s${SEED}" ;;
        sdftce|sdft) echo "dist_$1_perm$4_$3_v2_s${SEED}" ;;
    esac
}
prefix_of () { case $1 in tacred|fewrel) echo "$1_perm" ;; *) echo "$1_b10_perm" ;; esac; }
live () { pgrep -f -- "$1( |$)" > /dev/null; }   # the run name is on some process's command line
park () {
    local dst="${R}/_failed/$1_$(date +%Y%m%d_%H%M)"
    log "  moving partial ${R}/$1 -> ${dst}"
    [ "${DRY}" = "1" ] || mv "${R}/$1" "${dst}"
}
wait_gpu () {  # $1 = gpu: block until it has NEED_MB free (a snapshot, so slots also stagger)
    local free
    [ "${DRY}" = "1" ] && return 0
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -dc '0-9')
        [ "${free:-0}" -ge "${NEED_MB}" ] && return 0
        log "  gpu$1 has ${free:-?} MiB free < ${NEED_MB}, waiting"; sleep 120
    done
}
ensure_tokenized () {  # $1 = data prefix $2 = perm: base task data, as project_commands.sh step 2
    local pre=$1 p=$2 n t out
    n=$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "data/${pre}${p}/streams.json")
    for t in $(seq 0 $((n - 1))); do
        out="processed_data/${pre}${p}/${t}"
        [ -f "${out}/qwen/train_0.idx" ] && continue
        log "  tokenising ${pre}${p} task${t}"
        [ "${DRY}" = "1" ] && continue
        PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${pre}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${MODEL_PATH}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type qwen > "logs/tok_${pre}${p}_t${t}.log" 2>&1 || return 1
    done
}
results_dump () {  # $1 = run $2 = log prefix: per-task log.txt dump, as dist_queue.sh writes
    : > "$2_results.log"
    for f in $(find "${R}/$1" -name log.txt 2>/dev/null | sort -V); do
        echo "===== ${f#${R}/$1/} =====" >> "$2_results.log"; cat "${f}" >> "$2_results.log"
    done
}

# ---------------------------------------------------------------- GENEVA distillation perm3 (project_commands_4.sh)
geneva_perm3 () {  # $1 = gpu $2 = port: the seven methods one after another; 0 = all done
    local g=$1 port=$2 p=3 shared="dist_shared_task0_perm3_geneva_v2_s${SEED}" n=0 m kdt run pre rc
    # a shared task0 counts only with its model on disk; else dist_queue.sh trains it again
    if [ -e "${R}/${shared}" ] && { [ ! -f "${R}/${shared}/.complete" ] || [ ! -d "${R}/${shared}/task0/merged" ]; }; then
        live "run-name ${shared}" && { log "  ${shared} is training elsewhere; skip GENEVA perm3"; return 1; }
        park "${shared}"
    fi
    for m in rkl srkl distillm kd sfkl csd amid; do   # rkl first: dist_queue.sh trains the shared task0 before it
        run="dist_${m}_perm${p}_geneva_v2_s${SEED}"; pre="logs/geneva_dist_${m}_perm${p}"; rc=0
        if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; continue; fi
        if live "run-name ${run}"; then log "  skip ${run} (running elsewhere)"; continue; fi
        [ -e "${R}/${run}" ] && park "${run}"
        wait_gpu "${g}"
        log "  start ${run} on gpu${g}"
        [ "${DRY}" = "1" ] && continue
        case ${m} in
            rkl|srkl|distillm)   # dist_queue.sh's own batch sizes (16 x 2), as the other perms
                PERM="${p}" GPU="${g}" PROTOCOL=geneva_v2 SEED="${SEED}" DATA_PREFIX=geneva_b10_perm DIST_METHODS="${m}" \
                MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh >> "${LOG}" 2>&1 || rc=$? ;;
            *)                   # 8 x 4, as project_commands_3/4.sh ran kd/sfkl/csd/amid
                kdt=${m}; [ "${m}" = "amid" ] && kdt=adaptive-amid
                local extra=()
                [ "${m}" = "amid" ] && extra=(--extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5")
                MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh \
                    --run-name "${run}" --mode ce_kd --data-prefix geneva_b10_perm --perm "${p}" \
                    --kd-type "${kdt}" --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
                    --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 8 --acc 4 \
                    --greedy 1 --gpus "${g}" --start-task 1 --task0-source-run "${shared}" "${extra[@]}" \
                    > "${pre}_steps.log" 2>&1 || rc=$?
                results_dump "${run}" "${pre}" ;;
        esac
        if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; else
            log "  FAILED ${run} (exit ${rc}), see ${pre}_steps.log"; n=$((n + 1))
            [ -f "${R}/${shared}/.complete" ] || { log "  shared task0 perm3 did not finish; skip the rest of perm3"; return 1; }
        fi
    done
    [ "${n}" -eq 0 ]
}

run_job () {  # $1 gpu $2 port $3 job -> 0 done or skipped, 1 failed
    local g=$1 port=$2 kind name ds p rc=0 run pre jlog args
    IFS='|' read -r kind name ds p <<< "$3"
    if [ "${kind}" = "geneva3" ]; then geneva_perm3 "${g}" "${port}"; return; fi
    run=$(run_of "${kind}" "${name}" "${ds}" "${p}"); jlog="logs/p10_${run}.log"
    pre=$(prefix_of "${ds}"); [ "${name}" = "d_full" ] && pre=ace_b0_perm
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    [ -s "data/${pre}${p}/streams.json" ] || { log "  FAILED ${run}: missing data/${pre}${p}"; return 1; }
    if [ "${kind}" != "cllora_ours" ]; then   # task0 trains on the base split of the run's prefix
        ensure_tokenized "${pre}" "${p}" || { log "  FAILED ${run}: tokenising ${pre}${p}"; return 1; }
    fi
    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    case ${kind} in
        cllora_ours)   # project_commands_5.sh's command
            CUDA_VISIBLE_DEVICES="${g}" PYTHONPATH=. "${PY}" cl_lora/engine.py \
                --cl-method "${name}" --ours --data-root "data/ace_b10_perm${p}" --num-tasks 5 \
                --rank 16 --alpha 64 --lr 2e-4 --epochs 5 \
                --batch-size 8 --grad-accum 4 --eval-batch-size 16 \
                --seed "${SEED}" --model-path "${MODEL_PATH}" --save "${R}/${run}" > "${jlog}" 2>&1 || rc=$? ;;
        ace)           # ours_queue.sh's h12 + SD flags (project_commands.sh), task0 trained in the run
            args=(--run-name "${run}" --mode ce_kd --data-prefix ace_b10_perm --perm "${p}"
                  --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28"
                  --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16
                  --pl 1 --replay-boost 5 --greedy 1 --start-task 0 --gpus "${g}"
                  --pl-conf percentile --pl-conf-pct 70 --sd 1)
            case ${name} in
                m_omask) args+=(--sd-omask 1) ;;
                d_full)  args+=(--data-prefix ace_b0_perm --kd-scope pl) ;;
            esac
            MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" > "${jlog}" 2>&1 || rc=$? ;;
        sdftce|sdft)   # project_commands_6.sh's flags (SD_CE=1 / 0), micro-batch SD_MB, task0 trained in the run
            args=(--run-name "${run}" --mode ce_kd --data-prefix "${pre}" --perm "${p}"
                  --kd-type no --w-span 0 --pl 0
                  --sd 1 --w-sd 1.0 --sd-mu 0.99 --sd-temp 1.0 --sd-top-p 1.0 --sd-div fkl
                  --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs "${SD_MB}" --acc $((32 / SD_MB))
                  --greedy 1 --gpus "${g}" --start-task 0)
            if [ "${kind}" = "sdftce" ]; then
                args+=(--kd-ratio 0 --extra "--ced-sd-skip-tokens 3 --ced-sd-probe 0 --eval-interval -2")
            else
                args+=(--extra "--ced-sd-only --ced-sd-skip-tokens 3 --ced-sd-probe 0 --eval-interval -2")
            fi
            MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" > "${jlog}" 2>&1 || rc=$?
            results_dump "${run}" "logs/${ds}_dist_${kind}_perm${p}" ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog} and ${R}/${run}/task*/train.log"
    return 1
}

# Atomic claim, as in project_commands_6.sh: bash creates the file with O_EXCL. Not `mkdir`: the
# Rust coreutils mkdir of newer Ubuntu (26.04, uutils 0.8) lets two racing callers both succeed.
claim () { ( set -o noclobber; : > "$1" ) 2>/dev/null; }

worker () {  # $1 = gpu $2 = slot: take the next unclaimed job
    local g=$1 s=$2 port=$((30300 + 10 * $1 + $2)) j n=0 i=0
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))   # later slots see the earlier job's memory
    for j in "${JOBS[@]}"; do
        i=$((i + 1))
        claim "${CLAIMS}/${i}" || continue
        run_job "${g}" "${port}" "${j}" || n=$((n + 1))
    done
    log "worker gpu${g}/slot${s} finished, ${n} failed"
}

log "=== project_commands_10: steps '${STEPS}', ${#JOBS[@]} jobs, gpus ${GPUS[*]} x ${SLOTS} (DRY=${DRY}) ==="
pids=()
for s in $(seq 1 "${SLOTS}"); do
    for g in "${GPUS[@]}"; do worker "${g}" "${s}" & pids+=($!); done
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG} ==="
