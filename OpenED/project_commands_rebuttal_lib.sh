#!/usr/bin/env bash
# Rebuttal runs on ACE, no SD, 5 perms (joint: 3 seeds), started through project_commands_22.sh,
# 23.sh and 24.sh. One job per GPU, shortest jobs first (est_of below), so the longest run last.
# Every run trains its own task0. Run names start with r22_, except the memory runs, which take the
# names of their m = 10 siblings (ours_h2_f12_pl_..., cllora_...).
#   vplce   VPL + CE + replay x5 and x1 (Ours' pseudo-labels: dedup + lexicon, no confidence filter)
#   ce      CE + replay x5 and x1, no pseudo-labels
#   f12     Ours (f12_pl) at m = 10 again: the reported ACE runs (A40_3/f12_pl* on HF) kept no
#           final task4 answers and their processed test order is gone, so their errors cannot be
#           counted; this also times Ours at the paper's own setting
#   oracle  task targets keep the gold records of earlier tasks: CE + replay x5, and Ours (f12_pl)
#           without pseudo-labels
#   joint   one task with all 33 types and full annotation, plain CE, seeds 42 43 44
#   mem     memory m = 5 and 20: Ours (f12_pl) and IncLoRA / O-LoRA / TreeLoRA (32 x 1)
#   wc      CSD with WC_EPOCHS epochs on tasks 1-4 (task0 5 epochs): 2.41 / 0.75 s per batch
#           (tab:time, Ours w/o SD / CSD) puts Ours at ~3.2x CSD's time per epoch
#   q4b     Qwen3-4B (models/Qwen3-4B, download.txt): Ours (f12_pl), CSD and TreeLoRA, the best
#           baseline of each group on ACE. Same tokenizer as 0.6B, so the same tokenised data;
#           span layers 28 32 36 of 36 (0.6B: 22 25 28 of 28); physical batch 8 (the loss stays
#           per logical micro-batch), TreeLoRA 8 x 4 with the loss per 8 rows (see run_job).
#
# Data: ace_b10_perm* from ace_all.tar.gz; ace_b5_perm*, ace_b20_perm*, ace_joint_perm0 and the
# oracle split with is_memory on every train row (a negative row there can hold old-type gold
# only, which ced_oversample.py and the KD replay flag would otherwise take for a memory row) from
# ace_rebuttal.tar.gz. Both come from HF datht/processed-cl-ace via download.txt, already
# tokenised. Before any job the script unpacks what is missing (unpack_data): ace_b10_perm* from
# ace_all.tar.gz, then ace_rebuttal.tar.gz.
#
# Logs for later: the runner output of every job gets a timestamp per line (PL and training time
# per task), logs/<Q>_<run>_mem.csv samples the GPU's used memory every 2 s, and after each job
# its logs, test answers and test references (in the order of the answers) are copied to
# logs/rebuttal_runs/<run>/, which tools/ced_error_breakdown.py reads. The same is kept, at start, for
# finished runs of this host that the counts compare against (ACE g1_full and a_nosd, MAVEN Ours
# and IncLoRA). All of it ends up in logs/<Q>_logs.tar.gz.
#
#   GPUS="4 5 6 7" bash project_commands_rebuttal_lib.sh        # every job on these GPUs
#   DRY=1 GPUS="4" bash project_commands_rebuttal_lib.sh        # print the plan, train nothing
#   GPUS="4 5" STEPS="vplce ce" PERMS="0 1" bash project_commands_rebuttal_lib.sh
#
# Split over GPU groups: project_commands_22.sh, 23.sh and 24.sh run this with PART 1-3, each its
# own GPUs and queue name Q (claims, pool log, log archive). Every perm of a kind of run stays in
# one part (part_of below); the parts hold 40 / 30 / 30 % of the estimated GPU hours (est_of).
# The estimate is optimizer steps (from the split sizes, 5 epochs, memory rows x boost) times the
# seconds per batch of tab:time (B200, GENEVA; plain CE 0.6 s, which tab:time does not list),
# 2 min per task for pseudo-labels, 6 min per task for loading, merging and test answers, and
# x6.7 for Qwen3-4B (its parameters over 0.6B's). About 135 GPU hours in all, not measured.
#
# Knobs: GPUS (required), STEPS ["vplce ce f12 q4b oracle joint mem wc"], PERMS [0 1 2 3 4],
# PART [all], Q [rebuttal], NEED_MB free MiB a job waits for [60000], WC_EPOCHS [16], SEED [42],
# GEN_BACKEND [hf].
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
GPUS=(${GPUS:?set GPUS, e.g. GPUS=\"4 5 6 7\"})
STEPS=${STEPS:-"vplce ce f12 q4b oracle joint mem wc"}
PERMS=${PERMS:-"0 1 2 3 4"}
PART=${PART:-}
Q=${Q:-rebuttal}
NEED_MB=${NEED_MB:-60000}
WC_EPOCHS=${WC_EPOCHS:-16}
SEED=${SEED:-42}
export GEN_BACKEND=${GEN_BACKEND:-hf} COMPILE_GEN=0
R=results/qwen3/ced
KEEP=logs/rebuttal_runs
CLAIMS=logs/${Q}_claims
LOCKS=logs/locks
mkdir -p logs "${R}/_failed" "${LOCKS}" "${KEEP}"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/${Q}_pool.log
log () { echo "[${Q} $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---------------------------------------------------------------- environment (as project_commands_9.sh)
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

# ---------------------------------------------------------------- jobs
# Ours f12_pl: the flags of project_commands_9.sh (ours_queue.sh's h2), task0 trained in the run
BASE=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine
      --layers "22 25 28" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16
      --pl 1 --replay-boost 5 --greedy 1 --start-task 0)
CL_METHODS="inclora olora tree"
Q4B=models/Qwen3-4B
Q4B_LAYERS="28 32 36"
JOBS=()   # kind|a|perm
has () { [[ " ${STEPS} " == *" $1 "* ]]; }
for p in ${PERMS}; do
    has vplce && JOBS+=("vplce5||${p}" "vplce1||${p}")
    has ce && JOBS+=("ce5||${p}" "ce1||${p}")
    has f12 && JOBS+=("f12||${p}")
    has oracle && JOBS+=("oracle_ce||${p}" "oracle_f12||${p}")
    if has mem; then
        for b in 5 20; do
            JOBS+=("f12_mem|${b}|${p}")
            for m in ${CL_METHODS}; do JOBS+=("cl_mem|${b}:${m}|${p}"); done
        done
    fi
    has wc && JOBS+=("csd_wc||${p}")
    has q4b && JOBS+=("f12_4b||${p}" "csd_4b||${p}" "tree_4b||${p}")
done
has joint && for s in 42 43 44; do JOBS+=("joint|${s}|0"); done

key_of () {  # job -> kind as split and estimated: memory runs by method and m
    local kind a
    IFS='|' read -r kind a _ <<< "$1"
    case ${kind} in
        f12_mem) echo "f12_mem${a}" ;;
        cl_mem)  echo "${a#*:}_mem${a%%:*}" ;;
        *)       echo "${kind}" ;;
    esac
}
est_of () {  # kind -> estimated GPU minutes of one job (header)
    case $1 in
        joint) echo 11 ;;        ce1) echo 38 ;;           ce5|oracle_ce) echo 42 ;;
        inclora_mem5) echo 43 ;; inclora_mem20) echo 46 ;; vplce1) echo 46 ;;
        olora_mem5) echo 46 ;;   tree_mem5) echo 48 ;;     olora_mem20) echo 49 ;;
        vplce5) echo 50 ;;       tree_mem20) echo 51 ;;    csd_wc) echo 57 ;;
        f12_mem5) echo 73 ;;     oracle_f12) echo 76 ;;    f12) echo 83 ;;
        f12_mem20) echo 101 ;;   csd_4b) echo 121 ;;       tree_4b) echo 188 ;;
        f12_4b) echo 415 ;;
    esac
}
part_of () {  # kind -> part: 40 / 30 / 30 % of the estimated hours
    case $1 in
        joint|inclora_mem5|f12|f12_mem20|f12_4b)                                         echo 1 ;;
        inclora_mem20|vplce1|olora_mem5|tree_mem5|vplce5|csd_wc|f12_mem5|csd_4b)          echo 2 ;;
        ce1|ce5|oracle_ce|olora_mem20|tree_mem20|oracle_f12|tree_4b)                     echo 3 ;;
    esac
}
# this part's jobs, shortest kind first, perms in order within a kind
JOBS=($(for i in "${!JOBS[@]}"; do
            k=$(key_of "${JOBS[$i]}")
            [ -z "${PART}" ] || [ "$(part_of "${k}")" = "${PART}" ] || continue
            printf '%05d %s %05d %s\n' "$(est_of "${k}")" "${k}" "${i}" "${JOBS[$i]}"
        done | sort | cut -d' ' -f4))

run_of () {  # kind a perm -> run name
    case $1 in
        vplce5)     echo "r22_vplce_rep5_perm$3_ace_s${SEED}" ;;
        vplce1)     echo "r22_vplce_rep1_perm$3_ace_s${SEED}" ;;
        ce5)        echo "r22_ce_rep5_perm$3_ace_s${SEED}" ;;
        ce1)        echo "r22_ce_rep1_perm$3_ace_s${SEED}" ;;
        f12)        echo "r22_f12_pl_perm$3_ace_s${SEED}" ;;
        oracle_ce)  echo "r22_oracle_ce_rep5_perm$3_ace_s${SEED}" ;;
        oracle_f12) echo "r22_oracle_f12_perm$3_ace_s${SEED}" ;;
        joint)      echo "r22_joint_ace_s$2" ;;
        f12_mem)    echo "ours_h2_f12_pl_perm$3_ace_b$2_v2_s${SEED}" ;;
        cl_mem)     echo "cllora_${2#*:}_perm$3_ace_b${2%%:*}_v2_s${SEED}" ;;
        csd_wc)     echo "r22_csd_wc${WC_EPOCHS}_perm$3_ace_s${SEED}" ;;
        f12_4b)     echo "r22_f12_pl_perm$3_ace_qwen4b_s${SEED}" ;;
        csd_4b)     echo "r22_csd_perm$3_ace_qwen4b_s${SEED}" ;;
        tree_4b)    echo "cllora_tree_perm$3_ace_qwen4b_v2_s${SEED}" ;;
    esac
}
prefix_of () {  # kind a -> data prefix
    case $1 in
        oracle_*) echo ace_oracle_b10_perm ;;
        joint)    echo ace_joint_perm ;;
        f12_mem)  echo "ace_b$2_perm" ;;
        cl_mem)   echo "ace_b${2%%:*}_perm" ;;
        *)        echo ace_b10_perm ;;
    esac
}

# ---------------------------------------------------------------- helpers
live () { pgrep -f -- "$1( |$)" > /dev/null; }   # the run name is on some process's command line
park () {
    local dst="${R}/_failed/$1_$(date +%Y%m%d_%H%M)"
    log "  moving partial ${R}/$1 -> ${dst}"
    [ "${DRY}" = "1" ] || mv "${R}/$1" "${dst}"
}
wait_gpu () {  # $1 = gpu: block until it has NEED_MB free
    local free
    [ "${DRY}" = "1" ] && return 0
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -dc '0-9')
        [ "${free:-0}" -ge "${NEED_MB}" ] && return 0
        log "  gpu$1 has ${free:-?} MiB free < ${NEED_MB}, waiting"; sleep 120
    done
}
# Atomic claim: bash creates the file with O_EXCL (not `mkdir`, not atomic in uutils on Ubuntu 26.04)
claim () { ( set -o noclobber; : > "$1" ) 2>/dev/null; }
lock () {  # as project_commands_lib.sh: the file holds the owner's pid, a dead owner's lock is broken
    local f="${LOCKS}/$1" pid
    until ( set -o noclobber; echo $$ > "${f}" ) 2>/dev/null; do
        pid=$(cat "${f}" 2>/dev/null)
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then rm -f "${f}"; continue; fi
        sleep 30
    done
}
unlock () { rm -f "${LOCKS}/$1"; }
stamp () { local l; while IFS= read -r l; do printf '[%(%F %T)T] %s\n' -1 "${l}"; done; }

data_ready () {  # $1 = data prefix $2 = perm: split and its tokenised tasks present (oracle: tagged)
    [ -s "data/$1$2/streams.json" ] && [ -f "processed_data/$1$2/0/qwen/train_0.idx" ] || return 1
    [ "$1" != "ace_oracle_b10_perm" ] || grep -q '"is_memory"' "data/$1$2/1/train.jsonl"
}
all_ready () {  # $@ = data prefixes, perms 0-4 (joint: 0)
    local pre p
    for pre in "$@"; do
        for p in 0 1 2 3 4; do
            [ "${pre}" = ace_joint_perm ] && [ "${p}" != 0 ] && continue
            data_ready "${pre}" "${p}" || return 1
        done
    done
}
unpack_data () {  # once, before any job: the archives download.txt puts here, under one lock
    # (22/23/24 share this directory). ace_b10 first and only its own members: ace_all.tar.gz
    # also holds an untagged oracle split, which must not overwrite the tagged one
    lock unpack_ace
    if all_ready ace_b10_perm; then log "data: ace_b10_perm0-4 already unpacked"
    elif [ ! -e ace_all.tar.gz ]; then log "data: no ace_all.tar.gz here, fetch it with download.txt"
    elif [ "${DRY}" = "1" ]; then log "data: would unpack ace_b10_perm* from ace_all.tar.gz"
    else
        log "data: unpacking ace_b10_perm* from ace_all.tar.gz"
        tar xzf ace_all.tar.gz --wildcards 'data/ace_b10_perm*' 'processed_data/ace_b10_perm*' >> "${LOG}" 2>&1 \
            || log "data: unpacking ace_all.tar.gz failed"
    fi
    if all_ready ace_b5_perm ace_b20_perm ace_oracle_b10_perm ace_joint_perm; then
        log "data: ace_rebuttal splits already unpacked"
    elif [ ! -e ace_rebuttal.tar.gz ]; then log "data: no ace_rebuttal.tar.gz here, fetch it with download.txt"
    elif [ "${DRY}" = "1" ]; then log "data: would unpack ace_rebuttal.tar.gz"
    else
        log "data: unpacking ace_rebuttal.tar.gz"
        tar xzf ace_rebuttal.tar.gz >> "${LOG}" 2>&1 || log "data: unpacking ace_rebuttal.tar.gz failed"
    fi
    unlock unpack_ace
}
ensure_data () {  # $1 = data prefix $2 = perm
    [ "${DRY}" = "1" ] || data_ready "$1" "$2"
}
ensure_tokenized () {  # $1 = data prefix $2 = perm: base task data (normally shipped tokenised)
    local pre=$1 p=$2 n t out rc=0
    [ "${DRY}" = "1" ] && return 0
    n=$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "data/${pre}${p}/streams.json") || return 1
    lock "tok_${pre}${p}_qwen"
    for t in $(seq 0 $((n - 1))); do
        out="processed_data/${pre}${p}/${t}"
        [ -f "${out}/qwen/train_0.idx" ] && continue
        log "  tokenising ${pre}${p} task${t}"
        PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${pre}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${MODEL_PATH}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type qwen > "logs/tok_${pre}${p}_t${t}_qwen.log" 2>&1 || { rc=1; break; }
    done
    unlock "tok_${pre}${p}_qwen"
    return "${rc}"
}
keep_logs () {  # $1 = run: its logs, test answers and test references, small, to ${KEEP}/<run>
    local run=$1
    [ -d "${R}/${run}" ] || return 0
    cp "logs/${Q}_${run}.log" "logs/${Q}_${run}_mem.csv" "${KEEP}/" 2> /dev/null
    ( cd "${R}" && find "${run}" \( -name log.txt -o -name args.json -o -name answers.jsonl -o -name 'pl_task*.log' \
        -o -name run_config.txt -o -name run_manifest.json -o -name cl_results.json -o -path '*/predictions/*' \) \
        -type f -print0 | xargs -0 -r cp --parents -t "${HERE}/${KEEP}" )
    # the answers follow the order of the processed test.jsonl the run read (process_data.py
    # shuffles it), so keep that file next to them
    "${PY}" - "${R}/${run}" "${KEEP}/${run}" <<'EOF'
import glob, json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
for a in glob.glob(os.path.join(src, "task*", "**", "args.json"), recursive=True):
    ref = os.path.join(json.load(open(a))["data_dir"], "test.jsonl")
    if os.path.exists(ref):
        out = os.path.join(dst, os.path.relpath(os.path.dirname(a), src), "test.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy(ref, out)
EOF
}

# ---------------------------------------------------------------- one job
run_job () {  # $1 gpu $2 port $3 job -> 0 done or skipped, 1 failed
    local g=$1 port=$2 kind a p rc=0 run pre jlog args mpid
    IFS='|' read -r kind a p <<< "$3"
    run=$(run_of "${kind}" "${a}" "${p}"); pre=$(prefix_of "${kind}" "${a}"); jlog="logs/${Q}_${run}.log"
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    ensure_data "${pre}" "${p}" || { log "  FAILED ${run}: no data/${pre}${p} (see the data: lines at start)"; return 1; }
    if [ "${kind}" != "cl_mem" ] && [ "${kind}" != "tree_4b" ]; then
        ensure_tokenized "${pre}" "${p}" || { log "  FAILED ${run}: tokenising ${pre}${p}"; return 1; }
    fi
    if [[ ${kind} == *_4b ]] && [ ! -f "${Q4B}/config.json" ] && [ "${DRY}" != "1" ]; then
        log "  FAILED ${run}: no ${Q4B} (download.txt fetches it)"; return 1
    fi
    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -i "${g}" -lms 2000 \
        > "logs/${Q}_${run}_mem.csv" 2> /dev/null & mpid=$!
    args=(--run-name "${run}" --data-prefix "${pre}" --perm "${p}" "${BASE[@]}" --gpus "${g}")
    case ${kind} in   # flags after BASE, so they win
        vplce5)     args+=(--mode sft) ;;
        vplce1)     args+=(--mode sft --replay-boost 1) ;;
        ce5)        args+=(--mode sft --pl 0) ;;
        ce1)        args+=(--mode sft --pl 0 --replay-boost 1) ;;
        oracle_ce)  args+=(--mode sft --pl 0) ;;
        oracle_f12) args+=(--pl 0) ;;
        joint)      args+=(--mode sft --pl 0 --replay-boost 1 --seed "${a}" --end-task 0) ;;
        f12_4b)     args+=(--layers "${Q4B_LAYERS}") ;;
        f12|f12_mem) ;;
    esac
    case ${kind} in
        cl_mem)   # run_all_cllora.sh's flags, as project_commands_lib.sh's cl
            bash scripts/qwen/ced/run_cllora.sh --method "${a#*:}" --data-root "data/${pre}${p}" --num-tasks 5 \
                --rank 16 --alpha 64 --lr 2e-4 --epochs 5 --batch-size 32 --grad-accum 1 --eval-batch-size 16 \
                --protocol "ace_b${a%%:*}_v2" --seed "${SEED}" --gpu "${g}" --py "${PY}" 2>&1 | stamp > "${jlog}" || rc=$? ;;
        csd_wc)   # dist_queue.sh's CSD (32 x 1, no PL, no boost); task0 5 epochs, tasks 1-4 WC_EPOCHS
            args=(--run-name "${run}" --mode ce_kd --data-prefix "${pre}" --perm "${p}" --kd-type csd --w-span 0
                  --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" --rank 16 --alpha 64
                  --lr 0.0002 --seed "${SEED}" --bs 32 --acc 1 --greedy 1 --gpus "${g}")
            MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" --epochs 5 --start-task 0 --end-task 0 \
                2>&1 | stamp > "${jlog}" || rc=$?
            if [ "${rc}" -eq 0 ] && [ -d "${R}/${run}/task0/merged" ]; then
                rm -f "${R}/${run}/.complete"   # written after task0 only
                MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" --epochs "${WC_EPOCHS}" \
                    --start-task 1 --resume 2>&1 | stamp >> "${jlog}" || rc=$?
            fi ;;
        f12_4b)   # Ours on Qwen3-4B, physical batch 8
            MODEL_PATH="${Q4B}" PHYS_BS=8 GRAD_CKPT=1 MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" \
                2>&1 | stamp > "${jlog}" || rc=$? ;;
        csd_4b)   # dist_queue.sh's CSD (32 x 1 logical, no PL, no boost) on Qwen3-4B, task0 trained in the run
            args=(--run-name "${run}" --mode ce_kd --data-prefix "${pre}" --perm "${p}" --kd-type csd --w-span 0
                  --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "${Q4B_LAYERS}" --rank 16 --alpha 64
                  --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 32 --acc 1 --greedy 1 --gpus "${g}" --start-task 0)
            MODEL_PATH="${Q4B}" PHYS_BS=8 MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" \
                2>&1 | stamp > "${jlog}" || rc=$? ;;
        tree_4b)  # run_all_cllora.sh's TreeLoRA on Qwen3-4B as 8 x 4 (32 per update). PHYS_BS cannot
                  # split 32 x 1 (phys_split only merges) and the engine has no gradient checkpointing,
                  # so 32 rows at once on 4B is the likely cause of its exit 1; the loss is a token mean per 8 rows
            MODEL_PATH="${Q4B}" bash scripts/qwen/ced/run_cllora.sh --method tree --data-root "data/${pre}${p}" \
                --num-tasks 5 --rank 16 --alpha 64 --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4 \
                --eval-batch-size 16 --protocol ace_qwen4b_v2 --seed "${SEED}" --gpu "${g}" --py "${PY}" \
                2>&1 | stamp > "${jlog}" || rc=$? ;;
        *)        # no SD: PHYS_BS 32 + GRAD_CKPT, as project_commands_lib.sh's ours
            PHYS_BS=32 GRAD_CKPT=1 MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" \
                2>&1 | stamp > "${jlog}" || rc=$? ;;
    esac
    kill "${mpid}" 2> /dev/null
    keep_logs "${run}"
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog} and ${R}/${run}/task*/train.log"
    return 1
}

worker () {  # $1 = gpu: take the next unclaimed job
    local g=$1 port=$((32000 + 10 * $1)) j n=0 i=0
    for j in "${JOBS[@]}"; do
        i=$((i + 1))
        claim "${CLAIMS}/${i}" || continue
        run_job "${g}" "${port}" "${j}" || n=$((n + 1))
    done
    log "worker gpu${g} finished, ${n} failed"
}

HERE=$(pwd)
log "=== ${Q} (rebuttal lib, part ${PART:-all}): steps '${STEPS}', perms '${PERMS}', ${#JOBS[@]} jobs, gpus ${GPUS[*]}, gen ${GEN_BACKEND} (DRY=${DRY}) ==="
unpack_data
# finished runs of this host to count next to the new ones: Ours with SD (g1_full) and its -SD
# ablation (a_nosd) on ACE, Ours and IncLoRA on MAVEN (highest stripping); skipped where absent
if [ "${DRY}" != "1" ]; then
    for d in ${R}/ours_h12_sd_g1_full_perm*_ace_v2_s42 ${R}/ours_h12_a_nosd_perm*_ace_v2_s42 \
             ${R}/ours_h2_f12_pl_perm*_maven_v2_s42 ${R}/cllora_inclora_perm*_maven_v2_s42; do
        [ -f "${d}/.complete" ] && keep_logs "$(basename "${d}")" && log "  kept logs of $(basename "${d}")"
    done
fi
pids=()
for g in "${GPUS[@]}"; do worker "${g}" & pids+=($!); done
wait "${pids[@]}"
[ "${DRY}" = "1" ] || tar czf "logs/${Q}_logs.tar.gz" "${KEEP}" "logs/${Q}"_*.log "logs/${Q}"_*_mem.csv 2>> "${LOG}"
log "=== all done: grep FAILED ${LOG}; logs in logs/${Q}_logs.tar.gz ==="
