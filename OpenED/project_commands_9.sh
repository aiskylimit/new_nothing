#!/usr/bin/env bash
# Runs the paper tables still need and no other queue on this host covers, on GPUs 6 and 7.
#   ours        Ours (f12_pl) on GENEVA, RAMS and MAVEN, 5 perms each: the Ours row of Tables 1-2
#   ablation    ACE, 5 perms each, the configs (and run names) of project_commands.sh:
#                 g4_nospan                      ablation table, -span KD
#                 g2_ground g2_nofilter          ablation table, PL filter: grounding only, unfiltered
#                 l_olora l_inclora l_tree d_full   memory table, m = 0
#                 g4_cka g3_wsd30 g3_wsd03 g3_wsd01 sensitivity table (span metric, SD weight)
#
#   bash project_commands_9.sh                             # both steps, gpus 6 and 7
#   DRY=1 bash project_commands_9.sh                       # print the plan, train nothing
#   STEPS="ours" DATASETS="geneva" bash project_commands_9.sh
#
# Every run trains its own task0 (plain CE, as the shared task0 runs do) and goes on from it, so
# nothing here reads a checkpoint that an earlier run left on disk. CL-LoRA runs always did.
#
# Ours = f12_pl, the recipe of the ACE Ours row (ours_queue.sh's OURS_VARIANT=h2): PL with conflict
# dedup and the lexicon filter, no confidence filter, replay rows x5; 0.1 CE + 0.9 (SFKL skew 0.1
# + 2 span loss, layers 22 25 28, cosine) on replay rows; no SD. Run names:
# ours_h2_f12_pl_perm<p>_<ds>_v2_s42. The ACE configs take ours_queue.sh's h12 flags (PL
# confidence filter, top 70%) plus their own, as project_commands.sh runs them: g3/g4/d_full on
# the SD base (like g1_full), g2 on CE + PL only.
#
# project_commands.sh's ACE pool lists the same ACE configs after m_omask, under the same names.
# Both can run at once: a finished run is skipped, a run whose name is on a live process's command
# line is left to that process, and a crashed partial one is moved to results/qwen3/ced/_failed/
# (never deleted) and retrained. This file takes them roughly in the reverse of that pool's
# order, so the two meet late; when the pool reaches a run this file is training, ours_queue.sh
# refuses it ("run exists") and the pool logs it as FAILED, nothing more. run_config.txt says
# which kind of task0 a run used (task0_source=none: its own).
#
# Knobs: GPUS ["6 7"], SLOTS jobs per GPU [1; the ACE pool also uses 6 and 7], NEED_MB free MiB a
# job waits for [60000], STEPS ["ours ablation"], DATASETS for Ours ["geneva rams maven"],
# PERMS [0 1 2 3 4], SEED [42].
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
GPUS=(${GPUS:-6 7})
SLOTS=${SLOTS:-1}
NEED_MB=${NEED_MB:-60000}
STEPS=${STEPS:-"ours ablation"}
DATASETS=${DATASETS:-"geneva rams maven"}
PERMS=${PERMS:-"0 1 2 3 4"}
SEED=${SEED:-42}
R=results/qwen3/ced
CLAIMS=logs/p9_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/p9_pool.log
log () { echo "[p9 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---------------------------------------------------------------- environment (as project_commands.sh)
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then set +u; source "${VENV}/bin/activate"; set -u; fi
PY=${PY:-$(command -v python || command -v python3)}
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
if [ -f .env ] && [ -z "${HF_TOKEN:-}" ]; then set -a; . ./.env; set +a; fi   # building ace_b0_perm1-4
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
if [ -f models/Qwen3-0.6B/config.json ]; then
    MODEL_PATH=models/Qwen3-0.6B
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

# ---------------------------------------------------------------- jobs
# the flags ours_queue.sh passes run_ced_v2.sh, with task0 trained in the run instead of copied
BASE=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine
      --layers "22 25 28" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16
      --pl 1 --replay-boost 5 --greedy 1 --start-task 0)
# kind | config name | sd | runner flags | SD_ARGS   (kind: ours = f12_pl, abl = ACE config, cl = CL-LoRA)
ABLATIONS=(
  "abl|g4_nospan|1|--w-span 0|"
  "abl|g2_ground|0|--mode sft --pl 1 --pl-conf none --pl-lexicon 0|"
  "abl|g2_nofilter|0|--mode sft --pl 1 --pl-dedup 0 --pl-conf none --pl-lexicon 0|"
  "cl|l_olora|cl||"
  "cl|l_inclora|cl||"
  "cl|l_tree|cl||"
  "abl|d_full|1|--data-prefix ace_b0_perm --kd-scope pl|"
  "abl|g4_cka|1|--span-metric cka|"
  "abl|g3_wsd30|1||--w-sd 3.0"
  "abl|g3_wsd03|1||--w-sd 0.3"
  "abl|g3_wsd01|1||--w-sd 0.1"
)
JOBS=()   # kind|name|sd|flags|sd_args|dataset|perm, in hand-out order
if [[ " ${STEPS} " == *" ours "* ]]; then
    for ds in ${DATASETS}; do for p in ${PERMS}; do JOBS+=("ours|f12_pl|0|||${ds}|${p}"); done; done
fi
if [[ " ${STEPS} " == *" ablation "* ]]; then
    for c in "${ABLATIONS[@]}"; do for p in ${PERMS}; do JOBS+=("${c}|ace|${p}"); done; done
fi

run_of () {  # $1 kind $2 name $3 sd $4 dataset $5 perm -> run name (project_commands.sh's run_dir)
    case $1 in
        ours) echo "ours_h2_f12_pl_perm$5_$4_v2_s${SEED}" ;;
        cl)   echo "cllora_${2#l_}_perm$5_ace_b0_v2_s${SEED}" ;;
        abl)  local tag=""; [ "$3" = "1" ] && tag="_sd"; echo "ours_h12${tag}_$2_perm$5_ace_v2_s${SEED}" ;;
    esac
}
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
ensure_b0 () {  # $1 = perm: memory-0 ACE split, built as project_commands.sh step 2 does
    [ -s "data/ace_b0_perm$1/streams.json" ] && return 0
    [ -f data/ace/0/train.jsonl ] || { log "  no data/ace raw splits to build ace_b0_perm$1 from"; return 1; }
    log "  building data/ace_b0_perm$1 (--cap 0)"
    [ "${DRY}" = "1" ] && return 0
    claim data/ace_b0_build.lock || {   # another slot is building one, wait for it
        while [ -e data/ace_b0_build.lock ]; do sleep 30; done
        [ -s "data/ace_b0_perm$1/streams.json" ] && return 0
        claim data/ace_b0_build.lock || return 1
    }
    OPENED_BASE=$(pwd) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${PY}" tools/build_ced_perms.py \
        --cap 0 --perms "$1" --out-prefix ace_b0_perm >> "${LOG}" 2>&1
    local rc=$?
    rm -f data/ace_b0_build.lock
    [ "${rc}" -eq 0 ] && [ -s "data/ace_b0_perm$1/streams.json" ]
}

run_job () {  # $1 gpu $2 port $3 job -> 0 done or skipped, 1 failed
    local g=$1 port=$2 kind name sd flags sd_args ds p rc=0 run pre jlog args
    IFS='|' read -r kind name sd flags sd_args ds p <<< "$3"
    run=$(run_of "${kind}" "${name}" "${sd}" "${ds}" "${p}"); jlog="logs/p9_${run}.log"
    pre="${ds}_b10_perm"; [[ " ${flags} " == *" --data-prefix ace_b0_perm "* ]] && pre=ace_b0_perm
    [ "${kind}" = "cl" ] && pre=ace_b0_perm
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    if [ "${pre}" = "ace_b0_perm" ]; then
        ensure_b0 "${p}" || { log "  FAILED ${run}: no data/ace_b0_perm${p}"; return 1; }
    fi
    [ -s "data/${pre}${p}/streams.json" ] || [ "${DRY}" = "1" ] \
        || { log "  FAILED ${run}: missing data/${pre}${p}"; return 1; }
    if [ "${kind}" != "cl" ]; then   # task0 trains on the base split of the run's own prefix
        ensure_tokenized "${pre}" "${p}" || { log "  FAILED ${run}: tokenising ${pre}${p}"; return 1; }
    fi
    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    case ${kind} in
        ours|abl)
            args=(--run-name "${run}" --data-prefix "${ds}_b10_perm" --perm "${p}" "${BASE[@]}" --gpus "${g}")
            if [ "${kind}" = "abl" ]; then
                args+=(--pl-conf percentile --pl-conf-pct 70)   # ours_queue.sh's h12
                # shellcheck disable=SC2206   # sd_args and flags hold several runner flags
                [ "${sd}" = "1" ] && args+=(--sd 1 ${sd_args})
                # shellcheck disable=SC2206
                args+=(${flags})                                 # last, so they win
            fi
            MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" > "${jlog}" 2>&1 || rc=$? ;;
        cl)
            bash scripts/qwen/ced/run_cllora.sh --method "${name#l_}" --data-root "data/ace_b0_perm${p}" \
                --protocol ace_b0_v2 --seed "${SEED}" --gpu "${g}" --py "${PY}" > "${jlog}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog} and ${R}/${run}/task*/train.log"
    return 1
}

# Atomic claim, as in project_commands_6.sh: bash creates the file with O_EXCL. Not `mkdir`: the
# Rust coreutils mkdir of newer Ubuntu (26.04, uutils 0.8) lets two racing callers both succeed.
claim () { ( set -o noclobber; : > "$1" ) 2>/dev/null; }

worker () {  # $1 = gpu $2 = slot: take the next unclaimed job
    local g=$1 s=$2 port=$((30100 + 10 * $1 + $2)) j n=0 i=0
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))   # later slots see the earlier job's memory
    for j in "${JOBS[@]}"; do
        i=$((i + 1))
        claim "${CLAIMS}/${i}" || continue
        run_job "${g}" "${port}" "${j}" || n=$((n + 1))
    done
    log "worker gpu${g}/slot${s} finished, ${n} failed"
}

log "=== project_commands_9: steps '${STEPS}', Ours on '${DATASETS}', perms '${PERMS}', ${#JOBS[@]} jobs, gpus ${GPUS[*]} x ${SLOTS} (DRY=${DRY}) ==="
pids=()
for s in $(seq 1 "${SLOTS}"); do
    for g in "${GPUS[@]}"; do worker "${g}" "${s}" & pids+=($!); done
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG} ==="
