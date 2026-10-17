#!/usr/bin/env bash
# The paper runs no other script covers (checked 08/10), on GPU 0:
#   cre        Ours (f12_pl) on TACRED and FewRel, 5 perms each: the Ours row of tab:cre_main and
#              tab:pertask_cre. Data <ds>_perm, as the CRE baselines (run_cre_dist.sh, H200 runs).
#   rams0      RAMS perm0 of the baselines that only have perms 1-4: the 8 CL-LoRA methods, RKL and
#              DistiLLM (n=4 in tab:cee_main and tab:pertask_cee).
#   backbones  Ours (f12_pl) on GENEVA with Llama-3.2-1B-Instruct and Gemma-3-1b-it, 5 perms each
#              (tab:backbones_geneva). Only the new_nothingnew_2 checkout (NEW2) has Llama/Gemma
#              support, so these train there and their results land in NEW2/results/qwen3/ced/.
#
#   bash project_commands_11.sh                        # all steps, gpu 0, one job at a time
#   DRY=1 bash project_commands_11.sh                  # print the plan, train nothing
#   STEPS="rams0 cre" bash project_commands_11.sh      # jobs are handed out in STEPS order
#   SLOTS=2 bash project_commands_11.sh                # two jobs on gpu 0
#
# Ours = f12_pl as in project_commands_9.sh (ours_queue.sh's OURS_VARIANT=h2, no SD), each run
# training its own task0. Run names: ours_h2_f12_pl_perm<p>_<ds>_v2_s42, and for the other
# backbones ours_h2_f12_pl_perm<p>_geneva_<llama1b|gemma1b>_v2_s42, the tag project_commands_7.sh
# gives their baselines. On Llama/Gemma the span-loss layers are 13 14 16 and 20 23 26 (the
# relative depths of Qwen's 22 25 28) and the micro-batch is 8 x 4, as their baselines.
# RAMS perm0 uses the flags of perms 1-4 (run.sh): CL-LoRA through run_all_cllora.sh's 32 x 1,
# RKL and DistiLLM through dist_queue.sh, which by protocol share one task0 per perm and train it
# when it is missing or has lost its model.
#
# A finished run is skipped, a run whose name is on a live process's command line is left alone,
# and a crashed partial one is moved to results/qwen3/ced/_failed/ (never deleted) and retrained.
#
# Knobs: GPUS ["0"], SLOTS jobs per GPU [1], NEED_MB free MiB a job waits for [80000],
# STEPS ["cre rams0 backbones"], PERMS for cre and backbones [0 1 2 3 4], SEED [42],
# NEW2 [the new_nothingnew_2 OpenED checkout next to this one].
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
GPUS=(${GPUS:-0})
SLOTS=${SLOTS:-1}
NEED_MB=${NEED_MB:-80000}
STEPS=${STEPS:-"cre rams0 backbones"}
PERMS=${PERMS:-"0 1 2 3 4"}
SEED=${SEED:-42}
HERE=$(pwd)
# .../<x>_new_nothing/OpenED -> .../<x>_new_nothingnew_2/OpenED
NEW2=${NEW2:-$(dirname "${HERE}")new_2/OpenED}
R=results/qwen3/ced
CLAIMS=${HERE}/logs/p11_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=${HERE}/logs/p11_pool.log
log () { echo "[p11 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

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
# the flags ours_queue.sh passes run_ced_v2.sh, with task0 trained in the run instead of copied
BASE=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine
      --layers "22 25 28" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16
      --pl 1 --replay-boost 5 --greedy 1 --start-task 0)
CLLORA_METHODS="inclora olora tree inflora epi migu gainlora_o gainlora_inf"
JOBS=()
for st in ${STEPS}; do
    case ${st} in
        cre)       for ds in tacred fewrel; do for p in ${PERMS}; do JOBS+=("cre|f12_pl|${ds}|${p}"); done; done ;;
        rams0)     JOBS+=("dist|rkl distillm|rams|0")
                   for m in ${CLLORA_METHODS}; do JOBS+=("cl|${m}|rams|0"); done ;;
        backbones) for mt in llama gemma; do for p in ${PERMS}; do JOBS+=("bb|${mt}|geneva|${p}"); done; done ;;
        *) echo "unknown step '${st}' (cre rams0 backbones)"; exit 1 ;;
    esac
done

run_of () {  # $1 kind $2 name $3 dataset $4 perm -> run name
    case $1 in
        cre) echo "ours_h2_f12_pl_perm$4_$3_v2_s${SEED}" ;;
        cl)  echo "cllora_$2_perm$4_rams_v2_s${SEED}" ;;
        bb)  echo "ours_h2_f12_pl_perm$4_geneva_${2}1b_v2_s${SEED}" ;;   # llama1b / gemma1b
    esac
}
live () { pgrep -f -- "$1( |$)" > /dev/null; }   # the run name is on some process's command line
park () {  # $1 = results dir $2 = run
    local dst="$1/_failed/$2_$(date +%Y%m%d_%H%M)"
    log "  moving partial $1/$2 -> ${dst}"
    [ "${DRY}" = "1" ] || { mkdir -p "$1/_failed"; mv "$1/$2" "${dst}"; }
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
ensure_tokenized () {  # $1 = repo dir $2 = data prefix $3 = perm $4 = model type $5 = model path: base task data
    local dir=$1 pre=$2 p=$3 mt=$4 mp=$5 n t out
    n=$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${dir}/data/${pre}${p}/streams.json") || return 1
    for t in $(seq 0 $((n - 1))); do
        out="processed_data/${pre}${p}/${t}"
        [ -f "${dir}/${out}/${mt}/train_0.idx" ] && continue
        log "  tokenising ${pre}${p} task${t} (${mt})"
        [ "${DRY}" = "1" ] && continue
        ( cd "${dir}" && PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${pre}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${mp}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type "${mt}" ) > "${HERE}/logs/tok_${pre}${p}_t${t}_${mt}.log" 2>&1 || return 1
    done
}

# ---------------------------------------------------------------- RAMS perm0, RKL and DistiLLM (dist_queue.sh)
rams0_dist () {  # $1 = gpu $2 = port: both methods from one shared task0; 0 = both done
    local g=$1 port=$2 shared="dist_shared_task0_perm0_rams_v2_s${SEED}" m run todo="" rc=0 n=0
    local jlog="logs/p11_rams_dist_perm0.log"
    for m in rkl distillm; do
        run="dist_${m}_perm0_rams_v2_s${SEED}"
        if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; continue; fi
        if live "${run}"; then log "  skip ${run} (running elsewhere)"; continue; fi
        [ -e "${R}/${run}" ] && park "${R}" "${run}"
        todo+=" ${m}"
    done
    [ -n "${todo}" ] || return 0
    # a shared task0 counts only with its model on disk; else dist_queue.sh trains it again
    if [ -e "${R}/${shared}" ] && { [ ! -f "${R}/${shared}/.complete" ] || [ ! -d "${R}/${shared}/task0/merged" ]; }; then
        live "run-name ${shared}" && { log "  ${shared} is training elsewhere; skip RAMS perm0 RKL/DistiLLM"; return 1; }
        park "${R}" "${shared}"
    fi
    [ -s data/rams_b10_perm0/streams.json ] || { log "  FAILED RAMS perm0 dist: missing data/rams_b10_perm0"; return 1; }
    ensure_tokenized . rams_b10_perm 0 qwen "${MODEL_PATH}" || { log "  FAILED RAMS perm0 dist: tokenising"; return 1; }
    wait_gpu "${g}"
    log "  start${todo} on RAMS perm0, gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    PERM=0 GPU="${g}" PROTOCOL=rams_v2 SEED="${SEED}" DATA_PREFIX=rams_b10_perm DIST_METHODS="${todo# }" \
    MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh > "${jlog}" 2>&1 || rc=$?
    for m in ${todo}; do
        run="dist_${m}_perm0_rams_v2_s${SEED}"
        if [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; else
            log "  FAILED ${run} (exit ${rc}), see ${jlog} and logs/rams_dist_${m}_perm0_steps.log"; n=$((n + 1))
        fi
    done
    [ "${n}" -eq 0 ]
}

run_job () {  # $1 gpu $2 port $3 job -> 0 done or skipped, 1 failed
    local g=$1 port=$2 kind name ds p rc=0 run root=${R} jlog args mpath layers
    IFS='|' read -r kind name ds p <<< "$3"
    if [ "${kind}" = "dist" ]; then rams0_dist "${g}" "${port}"; return; fi
    run=$(run_of "${kind}" "${name}" "${ds}" "${p}"); jlog="logs/p11_${run}.log"
    if [ "${kind}" = "bb" ]; then
        root=${NEW2}/${R}
        case ${name} in
            llama) mpath=models/Llama-3.2-1B-Instruct; layers="13 14 16" ;;
            gemma) mpath=models/gemma-3-1b-it;         layers="20 23 26" ;;
        esac
        [ -f "${NEW2}/chat_format.py" ] || { log "  FAILED ${run}: no Llama/Gemma checkout at NEW2=${NEW2}"; return 1; }
        [ -f "${NEW2}/${mpath}/config.json" ] || [ "${DRY}" = "1" ] \
            || { log "  FAILED ${run}: no ${NEW2}/${mpath} (project_commands_7.sh / 8.sh download it)"; return 1; }
    fi
    if [ -f "${root}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    case ${kind} in
        cre) [ -s "data/${ds}_perm${p}/streams.json" ] || { log "  FAILED ${run}: missing data/${ds}_perm${p}"; return 1; }
             ensure_tokenized . "${ds}_perm" "${p}" qwen "${MODEL_PATH}" \
                 || { log "  FAILED ${run}: tokenising ${ds}_perm${p}"; return 1; } ;;
        cl)  [ -s "data/rams_b10_perm${p}/streams.json" ] || { log "  FAILED ${run}: missing data/rams_b10_perm${p}"; return 1; } ;;
        bb)  [ -s "${NEW2}/data/geneva_b10_perm${p}/streams.json" ] \
                 || { log "  FAILED ${run}: missing ${NEW2}/data/geneva_b10_perm${p}"; return 1; }
             ensure_tokenized "${NEW2}" geneva_b10_perm "${p}" "${name}" "${mpath}" \
                 || { log "  FAILED ${run}: tokenising geneva_b10_perm${p} for ${name}"; return 1; } ;;
    esac
    [ -e "${root}/${run}" ] && park "${root}" "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    case ${kind} in
        cre)
            args=(--run-name "${run}" --data-prefix "${ds}_perm" --perm "${p}" "${BASE[@]}" --gpus "${g}")
            MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" > "${jlog}" 2>&1 || rc=$? ;;
        cl)   # run_all_cllora.sh's flags, as RAMS perms 1-4
            bash scripts/qwen/ced/run_cllora.sh --method "${name}" --data-root "data/rams_b10_perm${p}" --num-tasks 5 \
                --rank 16 --alpha 64 --lr 2e-4 --epochs 5 --batch-size 32 --grad-accum 1 --eval-batch-size 16 \
                --protocol rams_v2 --seed "${SEED}" --gpu "${g}" --py "${PY}" > "${jlog}" 2>&1 || rc=$? ;;
        bb)   # NEW2's runner reads MODEL_PATH / MODEL_TYPE (project_commands_7.sh exports the same)
            args=(--run-name "${run}" --data-prefix geneva_b10_perm --perm "${p}" "${BASE[@]}" --gpus "${g}"
                  --layers "${layers}" --bs 8 --acc 4)   # last, so they win
            ( cd "${NEW2}" && MODEL_PATH="${mpath}" MODEL_TYPE="${name}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" ) > "${jlog}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${root}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog} and ${root}/${run}/task*/train.log"
    return 1
}

# Atomic claim, as in project_commands_6.sh: bash creates the file with O_EXCL. Not `mkdir`: the
# Rust coreutils mkdir of newer Ubuntu (26.04, uutils 0.8) lets two racing callers both succeed.
claim () { ( set -o noclobber; : > "$1" ) 2>/dev/null; }

worker () {  # $1 = gpu $2 = slot: take the next unclaimed job
    local g=$1 s=$2 port=$((30500 + 10 * $1 + $2)) j n=0 i=0
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))   # later slots see the earlier job's memory
    for j in "${JOBS[@]}"; do
        i=$((i + 1))
        claim "${CLAIMS}/${i}" || continue
        run_job "${g}" "${port}" "${j}" || n=$((n + 1))
    done
    log "worker gpu${g}/slot${s} finished, ${n} failed"
}

log "=== project_commands_11: steps '${STEPS}', perms '${PERMS}', ${#JOBS[@]} jobs, gpus ${GPUS[*]} x ${SLOTS}, NEW2=${NEW2} (DRY=${DRY}) ==="
pids=()
for s in $(seq 1 "${SLOTS}"); do
    for g in "${GPUS[@]}"; do worker "${g}" "${s}" & pids+=($!); done
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG} ==="
