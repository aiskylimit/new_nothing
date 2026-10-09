#!/usr/bin/env bash
# Shared by project_commands_13.sh ... project_commands_18.sh: each of those sets Q (its name) and
# GPU (one card, never shared with another of them), sources this file, appends its jobs to JOBS
# (in order) and calls run_all. One run at a time on that card, so no queue OOMs another.
#
# Trainer of the h200-vllm branch (ced_step.py, ced_eval.py, gen_backend.py):
#   - GEN_BACKEND [auto]: vllm when gen_backend.find_vllm_python() finds the tested vLLM (0.27.x,
#     VLLM_PY / .venv-vllm / /venv/main) and its torch sees the GPU, else hf; the pool log says
#     which. vLLM writes the test answers and the pseudo-labels with the settings HF generate()
#     resolves (gen_backend.py), so decoding stays what the finished runs used. Sampling inside
#     training steps (SD, DistiLLM, AMiD) stays HF, eager (COMPILE_GEN=0).
#   - answers once per task, after its last update (run_ced_v2.sh's EVAL_GEN_MODE=final).
#   - the loss stays per logical micro-batch; PHYS_BS only groups rows on the GPU. PHYS_BS 32
#     (+ GRAD_CKPT for Ours) where project_commands_12.sh ran it: sentence-level CRE, and Ours
#     without SD. Runs with SD (SDFT-CE, the SD ablations) and the distillation baselines of
#     dist_queue.sh keep their physical batch: they peaked near the card's size before.
# Flags, batch sizes and run names are those of the queue each run comes from (project_commands_9
# to 12.sh), so a run here matches its finished siblings and those queues skip what finishes here.
# A finished run is skipped, a run whose name is on a live process's command line is left alone,
# and a crashed or killed partial one is moved to results/qwen3/ced/_failed/ (never deleted) and
# retrained from task0. Work two queues could do at once (unpacking an archive, tokenising a split,
# training a shared task0) takes a lock under logs/locks/ first; a lock whose owner died is broken.
#
# Jobs (kind|a|b|c):
#   sent_ours|<ds>|<p>       Ours f12_pl, sentence-level CRE (ds tacred|fewrel), from the shared task0
#   sent_dist|<ds>|<m>|<p>   FKL RKL SFKL SRKL CSD DistiLLM AMiD (fkl rkl sfkl srkl csd distillm amid), same
#   sent_cl|<ds>|<m>|<p>     CL-LoRA method <m>, sentence-level CRE
#   ours|<ds>|<p>            Ours f12_pl with its own task0 (tacred/fewrel = the old <ds>_perm data)
#   qdist|<ds>|<m>|<p>       a distillation baseline from dist_shared_task0_perm<p>_<ds>_v2_s42
#                            (dist_queue.sh; amid as project_commands_10.sh ran GENEVA)
#   cl|<ds>|<m>|<p>          CL-LoRA, as run_all_cllora.sh (32 x 1)
#   sdftce|<ds>|<p>          SDFT + CE (project_commands_10.sh), own task0
#   abl|<config>|<p>         ACE ablation config of project_commands_9.sh, own task0
#   bb|<llama|gemma>|<p>     Ours on GENEVA with Llama-3.2-1B / Gemma-3-1b, in the new_nothingnew_2
#                            checkout (NEW2, its own older trainer: only it supports these models)
#
# Knobs: DRY [0] (print the plan), NEED_MB free MiB a run waits for [60000], GEN_BACKEND [auto],
# SEED [42], NEW2 [the new_nothingnew_2 OpenED checkout next to this one].

DRY=${DRY:-0}
NEED_MB=${NEED_MB:-60000}
SEED=${SEED:-42}
PORT=$((31000 + 10 * GPU))
HERE=$(pwd)
NEW2=${NEW2:-$(dirname "${HERE}")new_2/OpenED}
R=results/qwen3/ced
LOCKS=logs/locks
mkdir -p logs "${R}/_failed" "${LOCKS}"
LOG=logs/${Q}_pool.log
log () { echo "[${Q} $(date '+%F %T')] $*" | tee -a "${LOG}"; }

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

# ---------------------------------------------------------------- generation backend (scripts/qwen/lib.sh's test)
GEN_BACKEND=${GEN_BACKEND:-auto}
if [ "${GEN_BACKEND}" = "auto" ]; then
    if CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c "import sys; from gen_backend import find_vllm_python, vllm_supported, \
vllm_sees_gpu; py, version = find_vllm_python(); sys.exit(0 if vllm_supported(version) and vllm_sees_gpu(py) else 1)" \
            > /dev/null 2>&1; then
        GEN_BACKEND=vllm
    else
        GEN_BACKEND=hf
    fi
fi
export GEN_BACKEND COMPILE_GEN=0

# ---------------------------------------------------------------- helpers
live () { pgrep -f -- "$1( |$)" > /dev/null; }   # the run name is on some process's command line
park () {  # $1 = results dir $2 = run
    local dst="$1/_failed/$2_$(date +%Y%m%d_%H%M)"
    log "  moving partial $1/$2 -> ${dst}"
    [ "${DRY}" = "1" ] || { mkdir -p "$1/_failed"; mv "$1/$2" "${dst}"; }
}
wait_gpu () {  # block until GPU has NEED_MB free
    local free
    [ "${DRY}" = "1" ] && return 0
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -dc '0-9')
        [ "${free:-0}" -ge "${NEED_MB}" ] && return 0
        log "  gpu${GPU} has ${free:-?} MiB free < ${NEED_MB}, waiting"; sleep 120
    done
}
t0_ready () { [ -f "${R}/$1/.complete" ] && [ -d "${R}/$1/task0/merged" ]; }
# Cross-queue lock: bash creates the file with O_EXCL (not `mkdir`: uutils mkdir on Ubuntu 26.04 is
# not atomic). The file holds the owner's pid, so a lock left by a killed queue is broken.
lock () {
    local f="${LOCKS}/$1" pid
    until ( set -o noclobber; echo $$ > "${f}" ) 2>/dev/null; do
        pid=$(cat "${f}" 2>/dev/null)
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then rm -f "${f}"; continue; fi
        sleep 30
    done
}
unlock () { rm -f "${LOCKS}/$1"; }
results_dump () {  # $1 = run $2 = log prefix: per-task log.txt dump, as dist_queue.sh writes
    : > "$2_results.log"
    for f in $(find "${R}/$1" -name log.txt 2>/dev/null | sort -V); do
        echo "===== ${f#${R}/$1/} =====" >> "$2_results.log"; cat "${f}" >> "$2_results.log"
    done
}
prefix_of () { case $1 in tacred|fewrel) echo "$1_perm" ;; *) echo "$1_b10_perm" ;; esac; }

ensure_tokenized () {  # $1 = repo dir $2 = data prefix $3 = perm $4 = model type $5 = model path: base task data
    local dir=$1 pre=$2 p=$3 mt=$4 mp=$5 n t out rc=0
    [ -s "${dir}/data/${pre}${p}/streams.json" ] || { log "  missing ${dir}/data/${pre}${p}"; return 1; }
    n=$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${dir}/data/${pre}${p}/streams.json") || return 1
    lock "tok_${pre}${p}_${mt}"
    for t in $(seq 0 $((n - 1))); do
        out="processed_data/${pre}${p}/${t}"
        [ -f "${dir}/${out}/${mt}/train_0.idx" ] && continue
        log "  tokenising ${pre}${p} task${t} (${mt})"
        [ "${DRY}" = "1" ] && continue
        ( cd "${dir}" && PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${pre}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${mp}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type "${mt}" ) > "${HERE}/logs/tok_${pre}${p}_t${t}_${mt}.log" 2>&1 || { rc=1; break; }
    done
    unlock "tok_${pre}${p}_${mt}"
    return "${rc}"
}

sent_ready () { [ -s "data/$1_sent_perm$2/streams.json" ] && [ -f "processed_data/$1_sent_perm$2/9/qwen/train_0.idx" ]; }
ensure_sent_data () {  # $1 = ds $2 = perm: unpack <ds>_all_new.tar.gz (download.txt puts it here), downloads nothing
    local tgz=$1_all_new.tar.gz out
    sent_ready "$1" "$2" && return 0
    [ "${DRY}" = "1" ] && { log "  would unpack ${tgz}"; return 0; }
    lock "unpack_$1"
    if ! sent_ready "$1" "$2"; then
        if [ -e "${tgz}" ]; then
            log "  unpacking ${PWD}/${tgz}"
            # only an archive holding data/<ds>_sent_perm*, never the old processed-cl one
            if out=$("${PY}" - "${tgz}" "$1_sent_perm" <<'EOF' 2>&1
import sys, tarfile
tgz, pre = sys.argv[1], sys.argv[2]
with tarfile.open(tgz) as tf:
    if not any(n.startswith(f"data/{pre}") for n in tf.getnames()):
        sys.exit(f"{tgz} holds no data/{pre}* (the old processed-cl archive?), not unpacking it")
    tf.extractall(".", filter="data")
EOF
            ); then log "  unpacked"; rm -f "${tgz}"; else log "  not unpacked: ${out}"; fi
        else
            log "  no ${PWD}/${tgz}: fetch it with download.txt"
        fi
    fi
    unlock "unpack_$1"
    sent_ready "$1" "$2"
}

ensure_sent_t0 () {  # $1 = ds $2 = perm: the shared plain-CE task0 of sentence-level CRE (project_commands_12.sh)
    local t0="dist_shared_task0_perm$2_$1_sent_s${SEED}" rc=0
    t0_ready "${t0}" && return 0
    lock "t0_${t0}"
    while live "run-name ${t0}"; do sleep 60; done
    if ! t0_ready "${t0}"; then
        [ -e "${R}/${t0}" ] && park "${R}" "${t0}"
        wait_gpu
        log "  start ${t0} on gpu${GPU}"
        if [ "${DRY}" != "1" ]; then
            ED_EVAL_ARG_PER_TYPE=1 PHYS_BS=32 MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${t0}" --mode sft \
                --data-prefix "$1_sent_perm" --perm "$2" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 \
                --seed "${SEED}" --bs 8 --acc 4 --eval-bs 64 --greedy 1 --gpus "${GPU}" --end-task 0 \
                > "logs/${Q}_${t0}.log" 2>&1 || rc=$?
            if t0_ready "${t0}"; then log "  done  ${t0}"; else log "  FAILED ${t0} (exit ${rc}), see logs/${Q}_${t0}.log"; fi
        fi
    fi
    unlock "t0_${t0}"
    t0_ready "${t0}" || [ "${DRY}" = "1" ]
}

# ACE ablation configs (project_commands_9.sh): sd | runner flags | SD flags
abl_of () {
    case $1 in
        g4_nospan)   echo "1|--w-span 0|" ;;
        g2_ground)   echo "0|--mode sft --pl 1 --pl-conf none --pl-lexicon 0|" ;;
        g2_nofilter) echo "0|--mode sft --pl 1 --pl-dedup 0 --pl-conf none --pl-lexicon 0|" ;;
        g4_cka)      echo "1|--span-metric cka|" ;;
        g3_wsd30)    echo "1||--w-sd 3.0" ;;
        g3_wsd03)    echo "1||--w-sd 0.3" ;;
        g3_wsd01)    echo "1||--w-sd 0.1" ;;
        *) return 1 ;;
    esac
}
SENT_DIST="fkl rkl sfkl srkl csd distillm amid"
SENT_CL="inclora olora tree inflora epi migu gainlora_o gainlora_inf"
ABL_CONFIGS="g2_ground g2_nofilter g4_nospan g4_cka g3_wsd30 g3_wsd03 g3_wsd01"
JOBS=()
sent_perm_jobs () {  # $1 = ds $2 = perm: every sentence-level CRE run of that perm (Ours, distillation, CL-LoRA)
    local m
    JOBS+=("sent_ours|$1|$2")
    for m in ${SENT_DIST}; do JOBS+=("sent_dist|$1|${m}|$2"); done
    for m in ${SENT_CL}; do JOBS+=("sent_cl|$1|${m}|$2"); done
}
# Ours f12_pl: the flags ours_queue.sh (OURS_VARIANT=h2) passes run_ced_v2.sh, task0 trained in the run
BASE=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine
      --layers "22 25 28" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16
      --pl 1 --replay-boost 5 --greedy 1 --start-task 0)

# ---------------------------------------------------------------- distillation from a shared task0 (dist_queue.sh)
qdist () {  # $1 = ds $2 = method $3 = perm
    local ds=$1 m=$2 p=$3 shared="dist_shared_task0_perm$3_$1_v2_s${SEED}" run="dist_$2_perm$3_$1_v2_s${SEED}"
    local pre="logs/$1_dist_$2_perm$3" rc=0
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "run-name ${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    ensure_tokenized . "${ds}_b10_perm" "${p}" qwen "${MODEL_PATH}" || { log "  FAILED ${run}: data ${ds}_b10_perm${p}"; return 1; }
    [ -e "${R}/${run}" ] && park "${R}" "${run}"
    lock "t0_${shared}"
    while live "run-name ${shared}"; do sleep 60; done
    if ! t0_ready "${shared}"; then   # counts only with its model on disk; dist_queue.sh trains it when missing
        [ -e "${R}/${shared}" ] && park "${R}" "${shared}"
        wait_gpu
        log "  start ${shared} on gpu${GPU}"
        [ "${DRY}" = "1" ] || PERM="${p}" GPU="${GPU}" PROTOCOL="${ds}_v2" SEED="${SEED}" DATA_PREFIX="${ds}_b10_perm" \
            DIST_METHODS="" MASTER_PORT="${PORT}" bash scripts/qwen/ced/dist_queue.sh >> "${LOG}" 2>&1
    fi
    unlock "t0_${shared}"
    t0_ready "${shared}" || [ "${DRY}" = "1" ] || { log "  FAILED ${run}: no shared task0 ${shared}"; return 1; }
    wait_gpu
    log "  start ${run} on gpu${GPU}"
    [ "${DRY}" = "1" ] && return 0
    case ${m} in
        amid)   # 8 x 4, as project_commands_3/4/10.sh ran it
            MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh \
                --run-name "${run}" --mode ce_kd --data-prefix "${ds}_b10_perm" --perm "${p}" \
                --kd-type adaptive-amid --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
                --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 8 --acc 4 \
                --greedy 1 --gpus "${GPU}" --start-task 1 --task0-source-run "${shared}" \
                --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5" \
                > "${pre}_steps.log" 2>&1 || rc=$?
            results_dump "${run}" "${pre}" ;;
        *)      # dist_queue.sh's own batch sizes, as the other perms
            PERM="${p}" GPU="${GPU}" PROTOCOL="${ds}_v2" SEED="${SEED}" DATA_PREFIX="${ds}_b10_perm" DIST_METHODS="${m}" \
            MASTER_PORT="${PORT}" bash scripts/qwen/ced/dist_queue.sh >> "${LOG}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${pre}_steps.log"
    return 1
}

# ---------------------------------------------------------------- one job
run_of () {  # kind a b c -> run name
    case $1 in
        sent_ours) echo "ours_h2_f12_pl_perm$3_$2_sent_s${SEED}" ;;
        sent_dist) echo "cre_sent_$2_$3_perm$4" ;;
        sent_cl)   echo "cllora_$3_perm$4_$2_sent_s${SEED}" ;;
        ours)      echo "ours_h2_f12_pl_perm$3_$2_v2_s${SEED}" ;;
        cl)        echo "cllora_$3_perm$4_$2_v2_s${SEED}" ;;
        sdftce)    echo "dist_sdftce_perm$3_$2_v2_s${SEED}" ;;
        abl)       local sd; sd=$(abl_of "$2" | cut -d'|' -f1); [ "${sd}" = "1" ] && echo "ours_h12_sd_$2_perm$3_ace_v2_s${SEED}" \
                       || echo "ours_h12_$2_perm$3_ace_v2_s${SEED}" ;;
        bb)        echo "ours_h2_f12_pl_perm$3_geneva_${2}1b_v2_s${SEED}" ;;   # llama1b / gemma1b
    esac
}

run_job () {  # $1 = job -> 0 done or skipped, 1 failed
    local k a b c run root=${R} jlog rc=0 pre args kdt extra sd flags sdargs mpath layers
    IFS='|' read -r k a b c <<< "$1"
    if [ "${k}" = "qdist" ]; then qdist "${a}" "${b}" "${c}"; return; fi
    run=$(run_of "${k}" "${a}" "${b}" "${c}"); jlog="logs/${Q}_${run}.log"
    [ -n "${run}" ] || { log "  FAILED unknown job '$1'"; return 1; }
    if [ "${k}" = "bb" ]; then
        root=${NEW2}/${R}
        case ${a} in
            llama) mpath=models/Llama-3.2-1B-Instruct; layers="13 14 16" ;;
            gemma) mpath=models/gemma-3-1b-it;         layers="20 23 26" ;;
        esac
    fi
    if [ -f "${root}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    case ${k} in   # data, and the shared task0 the run starts from
        sent_*)  ensure_sent_data "${a}" "${c:-${b}}" || { log "  FAILED ${run}: no data/${a}_sent_perm${c:-${b}}"; return 1; }
                 if [ "${k}" != "sent_cl" ]; then
                     ensure_sent_t0 "${a}" "${c:-${b}}" || { log "  FAILED ${run}: no shared task0"; return 1; }
                 fi ;;
        ours|sdftce)
                 pre=$(prefix_of "${a}")
                 ensure_tokenized . "${pre}" "${b}" qwen "${MODEL_PATH}" || { log "  FAILED ${run}: data ${pre}${b}"; return 1; } ;;
        abl)     ensure_tokenized . ace_b10_perm "${b}" qwen "${MODEL_PATH}" || { log "  FAILED ${run}: data ace_b10_perm${b}"; return 1; } ;;
        cl)      [ -s "data/${a}_b10_perm${c}/streams.json" ] || { log "  FAILED ${run}: missing data/${a}_b10_perm${c}"; return 1; } ;;
        bb)      [ -f "${NEW2}/chat_format.py" ] || { log "  FAILED ${run}: no Llama/Gemma checkout at NEW2=${NEW2}"; return 1; }
                 [ -f "${NEW2}/${mpath}/config.json" ] || [ "${DRY}" = "1" ] \
                     || { log "  FAILED ${run}: no ${NEW2}/${mpath} (project_commands_7.sh / 8.sh download it)"; return 1; }
                 ensure_tokenized "${NEW2}" geneva_b10_perm "${b}" "${a}" "${mpath}" \
                     || { log "  FAILED ${run}: tokenising geneva_b10_perm${b} for ${a}"; return 1; } ;;
    esac
    [ -e "${root}/${run}" ] && park "${root}" "${run}"
    wait_gpu
    log "  start ${run} on gpu${GPU} (gen ${GEN_BACKEND})"
    [ "${DRY}" = "1" ] && return 0
    case ${k} in
        sent_ours)   # project_commands_12.sh: ours_queue.sh's h2 flags, PL anchored on the (subject, object) pair
            ED_EVAL_ARG_PER_TYPE=1 PL_ANCHOR=pair PHYS_BS=32 GRAD_CKPT=1 MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh \
                --run-name "${run}" --mode ce_kd --data-prefix "${a}_sent_perm" --perm "${b}" --kd-type sfkl --w-span 2.0 \
                --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 \
                --seed "${SEED}" --bs 2 --acc 16 --eval-bs 64 --pl 1 --replay-boost 5 --greedy 1 --gpus "${GPU}" \
                --start-task 1 --task0-source-run "dist_shared_task0_perm${b}_${a}_sent_s${SEED}" > "${jlog}" 2>&1 || rc=$?
            [ -f "${R}/${run}/.complete" ] && rm -rf "${R}/${run}"/task*/merged ;;   # the last task's model, never read again
        sent_dist)   # project_commands_12.sh, run_cre_dist.sh's DistiLLM/AMiD flags
            case ${b} in
                distillm) kdt=adaptive-srkl; extra="--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000" ;;
                amid)     kdt=adaptive-amid; extra="--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5" ;;
                *)        kdt=${b};          extra="" ;;
            esac
            ED_EVAL_ARG_PER_TYPE=1 PHYS_BS=32 MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${run}" --mode ce_kd \
                --data-prefix "${a}_sent_perm" --perm "${c}" --kd-type "${kdt}" --w-span 0 --kd-ratio 0.9 --skew 0.1 \
                --span-metric cosine --layers "22 25 28" --rank 16 --alpha 64 --epochs 5 --lr 0.0002 \
                --seed "${SEED}" --bs 8 --acc 4 --eval-bs 64 --greedy 1 --gpus "${GPU}" --start-task 1 \
                --task0-source-run "dist_shared_task0_perm${c}_${a}_sent_s${SEED}" --extra "${extra}" > "${jlog}" 2>&1 || rc=$?
            [ -f "${R}/${run}/.complete" ] && rm -rf "${R}/${run}"/task*/merged ;;
        sent_cl)     # project_commands_12.sh
            ED_EVAL_ARG_PER_TYPE=1 PHYS_BS=32 bash scripts/qwen/ced/run_cllora.sh --method "${b}" --data-root "data/${a}_sent_perm${c}" \
                --num-tasks 10 --rank 16 --alpha 64 --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4 \
                --eval-batch-size 64 --protocol "${a}_sent" --seed "${SEED}" --gpu "${GPU}" --py "${PY}" > "${jlog}" 2>&1 || rc=$?
            rm -f "${R}/${run}/checkpoint_latest.pt" "${R}/${run}/checkpoint_latest.pt.tmp" ;;
        ours)        # project_commands_9/11.sh, task0 trained in the run
            PHYS_BS=32 GRAD_CKPT=1 MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${run}" \
                --data-prefix "${pre}" --perm "${b}" "${BASE[@]}" --gpus "${GPU}" > "${jlog}" 2>&1 || rc=$? ;;
        cl)          # run_all_cllora.sh's flags (project_commands_11.sh's RAMS perm0)
            bash scripts/qwen/ced/run_cllora.sh --method "${b}" --data-root "data/${a}_b10_perm${c}" --num-tasks 5 \
                --rank 16 --alpha 64 --lr 2e-4 --epochs 5 --batch-size 32 --grad-accum 1 --eval-batch-size 16 \
                --protocol "${a}_v2" --seed "${SEED}" --gpu "${GPU}" --py "${PY}" > "${jlog}" 2>&1 || rc=$? ;;
        sdftce)      # project_commands_10.sh: SDFT + CE, micro-batch 8 x 4, task0 trained in the run
            MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${run}" --mode ce_kd \
                --data-prefix "${pre}" --perm "${b}" --kd-type no --w-span 0 --pl 0 \
                --sd 1 --w-sd 1.0 --sd-mu 0.99 --sd-temp 1.0 --sd-top-p 1.0 --sd-div fkl \
                --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 8 --acc 4 \
                --greedy 1 --gpus "${GPU}" --start-task 0 \
                --kd-ratio 0 --extra "--ced-sd-skip-tokens 3 --ced-sd-probe 0 --eval-interval -2" > "${jlog}" 2>&1 || rc=$?
            results_dump "${run}" "logs/${a}_dist_sdftce_perm${b}" ;;
        abl)         # project_commands_9.sh: ours_queue.sh's h12 flags plus the config's own
            IFS='|' read -r sd flags sdargs <<< "$(abl_of "${a}")"
            args=(--run-name "${run}" --data-prefix ace_b10_perm --perm "${b}" "${BASE[@]}" --gpus "${GPU}"
                  --pl-conf percentile --pl-conf-pct 70)
            # shellcheck disable=SC2206   # sdargs and flags hold several runner flags
            [ "${sd}" = "1" ] && args+=(--sd 1 ${sdargs})
            # shellcheck disable=SC2206
            args+=(${flags})                                 # last, so they win
            if [ "${sd}" = "1" ]; then
                MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" > "${jlog}" 2>&1 || rc=$?
            else
                PHYS_BS=32 GRAD_CKPT=1 MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" > "${jlog}" 2>&1 || rc=$?
            fi ;;
        bb)          # project_commands_11.sh: NEW2's runner reads MODEL_PATH / MODEL_TYPE
            args=(--run-name "${run}" --data-prefix geneva_b10_perm --perm "${b}" "${BASE[@]}" --gpus "${GPU}"
                  --layers "${layers}" --bs 8 --acc 4)   # last, so they win
            ( cd "${NEW2}" && MODEL_PATH="${mpath}" MODEL_TYPE="${a}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                MASTER_PORT="${PORT}" bash scripts/qwen/ced/run_ced_v2.sh "${args[@]}" ) > "${jlog}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${root}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog} and ${root}/${run}/task*/train.log"
    return 1
}

run_all () {
    local j n=0
    log "=== ${Q}: ${#JOBS[@]} jobs on gpu${GPU}, gen ${GEN_BACKEND} (DRY=${DRY}) ==="
    for j in "${JOBS[@]}"; do run_job "${j}" || n=$((n + 1)); done
    log "=== ${Q} all done, ${n} failed: grep FAILED ${LOG} ==="
}
