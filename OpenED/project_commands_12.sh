#!/usr/bin/env bash
# Sentence-level CRE on TACRED: the model extracts the subject, the object and the relation
# itself, so the logs give entity F1 ("entity") next to triple F1 ("argument"). Every baseline
# plus Ours, 5 perms, on GPU 4 with several runs on it. project_commands_13.sh runs the same for
# FewRel (DS=fewrel).
#
#   bash project_commands_12.sh                       # TACRED, gpu 4, 2 runs at a time
#   DRY=1 bash project_commands_12.sh                 # print the plan, train nothing
#   STEPS="cllora" PERMS="0" bash project_commands_12.sh
#
# Data: datht/processed-new-cl-<ds>. download.txt puts <ds>_all_new.tar.gz here; this script only
# unpacks it (to data/<ds>_sent_perm<p> and processed_data/<ds>_sent_perm<p>), downloads nothing.
# Runs (run names as scripts/qwen/cre_sent/ in the local OpenED tree):
#   task0   dist_shared_task0_perm<p>_<ds>_sent_s42, plain CE, trained here once per perm; the
#           distillation baselines and Ours start from it (the paper's same theta_0)
#   dist    FKL RKL SFKL SRKL CSD DistiLLM AMiD (run_cre_dist.sh's flags): cre_sent_<ds>_<m>_perm<p>
#   cllora  IncLoRA O-LoRA TreeLoRA InfLoRA EPI MIGU GainLoRA(O-LoRA) GainLoRA(InfLoRA), task0
#           trained inside the engine: cllora_<m>_perm<p>_<ds>_sent_s42
#   ours    f12_pl (ours_queue.sh's h2: PL with dedup and lexicon filter, replay x5, SFKL + span
#           KD, no SD), PL anchored on the (subject, object) pair: ours_h2_f12_pl_perm<p>_<ds>_sent_s42
# ED_EVAL_ARG_PER_TYPE=1 adds per-relation triple counts (FGT on triples), PL_ANCHOR=pair the
# pair anchor; both only change these runs.
#
# Trainer of the h200-vllm branch (ced_step.py, ced_eval.py), same objectives as before:
#   - answers are generated once per task, for the test set after the last update
#     (run_ced_v2.sh's EVAL_GEN_MODE=final); DistiLLM and AMiD keep a dev loss pass for their
#     adaptive threshold. Evaluation batch 64, as the runs of 08/10.
#   - the loss stays defined per logical micro-batch: 8 x 4 for the baselines and task0 (as the
#     runs of 08/10), 2 x 16 for Ours (its recipe). PHYS_BS rows go through the GPU at once,
#     the loss is still taken per logical micro-batch (--loss-group-size), so only the speed
#     changes. Ours also recomputes activations in the backward (GRAD_CKPT=1) to fit 32 rows.
#   - Hugging Face generation (GEN_BACKEND=hf) and eager SD sampling (COMPILE_GEN=0): vLLM is not
#     checked on this host, and the compiled sampler broke Ours on H200.
#   - SLOTS runs at a time per GPU, each starting only when its GPU has NEED_MB free.
# Runs of 08/10 (perm 0, part of perms 1-2) were trained with the earlier trainer; their loss is
# the same per micro-batch, and they are skipped as finished.
# A finished run is skipped, a run whose name is on a live process's command line is left alone,
# and a crashed partial one is moved to results/qwen3/ced/_failed/ (never deleted) and retrained.
#
# Knobs: DS [tacred], GPUS ["4"], SLOTS runs per GPU [2], NEED_MB [50000], PHYS_BS [32],
# PERMS [0 1 2 3 4], STEPS ["cllora dist ours"], EPOCHS [5], SEED [42].
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
DS=${DS:-tacred}
GPUS=(${GPUS:-4})
SLOTS=${SLOTS:-2}
NEED_MB=${NEED_MB:-50000}
PHYS=${PHYS_BS:-32}
export GEN_BACKEND=hf COMPILE_GEN=0
PERMS=${PERMS:-"0 1 2 3 4"}
STEPS=${STEPS:-"cllora dist ours"}
EPOCHS=${EPOCHS:-5}
SEED=${SEED:-42}
case ${DS} in
    tacred) PORT0=30700 ;;
    fewrel) PORT0=30800 ;;
    *) echo "unknown DS '${DS}' (tacred|fewrel)"; exit 1 ;;
esac
PRE=${DS}_sent_perm
PROTO=${DS}_sent
R=results/qwen3/ced
CLAIMS=logs/p12_${DS}_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/p12_${DS}_pool.log
log () { echo "[p12 ${DS} $(date '+%F %T')] $*" | tee -a "${LOG}"; }
export ED_EVAL_ARG_PER_TYPE=1 PL_ANCHOR=pair

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

# ---------------------------------------------------------------- data
have_data () {
    local p
    for p in ${PERMS}; do
        [ -s "data/${PRE}${p}/streams.json" ] && [ -f "processed_data/${PRE}${p}/9/qwen/train_0.idx" ] || return 1
    done
}
unpack () {  # $1 = archive: only one holding data/<ds>_sent_perm*, never an old <ds>_perm one
    "${PY}" - "$1" "${PRE}" <<'EOF' 2>&1
import sys, tarfile
tgz, pre = sys.argv[1], sys.argv[2]
with tarfile.open(tgz) as tf:
    if not any(n.startswith(f"data/{pre}") for n in tf.getnames()):
        sys.exit(f"{tgz} holds no data/{pre}* (the old processed-cl archive?), not unpacking it")
    tf.extractall(".", filter="data")
EOF
}
if ! have_data; then
    log "data/${PRE}* or processed_data/${PRE}* missing"
    if [ "${DRY}" != "1" ]; then
        # download.txt puts the archive here, next to this script. Its own name, so the old
        # processed-cl <ds>_all.tar.gz can never stand in for it.
        TGZ=${DS}_all_new.tar.gz
        if [ -e "${TGZ}" ]; then
            log "  unpacking ${PWD}/${TGZ} ($(du -h "${TGZ}" | cut -f1))"
            if out=$(unpack "${TGZ}"); then
                log "  unpacked"; rm -f "${TGZ}"
            else
                log "  not unpacked: ${out}"
            fi
        else
            log "  no ${PWD}/${TGZ}"
        fi
        have_data || { log "no data/${PRE}*: fetch ${TGZ} with download.txt into ${PWD} and run again"; exit 1; }
    fi
fi

# ---------------------------------------------------------------- jobs: kind|name|perm
CLLORA_METHODS="inclora olora tree inflora epi migu gainlora_o gainlora_inf"
DIST_METHODS="fkl rkl sfkl srkl csd distillm amid"
JOBS=()   # hand-out order: per perm the shared task0, then everything else
for p in ${PERMS}; do
    [[ " ${STEPS} " == *" dist "* || " ${STEPS} " == *" ours "* ]] && JOBS+=("t0|task0|${p}")
    if [[ " ${STEPS} " == *" cllora "* ]]; then for m in ${CLLORA_METHODS}; do JOBS+=("cl|${m}|${p}"); done; fi
    if [[ " ${STEPS} " == *" dist "* ]]; then for m in ${DIST_METHODS}; do JOBS+=("dist|${m}|${p}"); done; fi
    [[ " ${STEPS} " == *" ours "* ]] && JOBS+=("ours|f12_pl|${p}")
done

run_of () {  # $1 kind $2 name $3 perm -> run name
    case $1 in
        t0)   echo "dist_shared_task0_perm$3_${PROTO}_s${SEED}" ;;
        cl)   echo "cllora_$2_perm$3_${PROTO}_s${SEED}" ;;
        dist) echo "cre_sent_${DS}_$2_perm$3" ;;
        ours) echo "ours_h2_f12_pl_perm$3_${PROTO}_s${SEED}" ;;
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
t0_ready () { [ -f "${R}/$1/.complete" ] && [ -d "${R}/$1/task0/merged" ]; }

run_job () {  # $1 gpu $2 port $3 kind $4 name $5 perm -> 0 done, 1 failed, 2 busy elsewhere
    local g=$1 port=$2 k=$3 m=$4 p=$5 rc=0 run t0 jlog kdt extra
    run=$(run_of "${k}" "${m}" "${p}"); t0=$(run_of t0 - "${p}"); jlog="logs/p12_${run}.log"
    if [ "${k}" = "t0" ]; then
        t0_ready "${run}" && { log "  skip ${run} (complete)"; return 0; }
        live "run-name ${run}" && return 2   # dist runs also name it, as --task0-source-run
    else
        [ -f "${R}/${run}/.complete" ] && { log "  skip ${run} (complete)"; return 0; }
        live "${run}" && return 2
    fi
    case ${k} in
        dist|ours) t0_ready "${t0}" || [ "${DRY}" = "1" ] || { log "  FAILED ${run}: no shared task0 ${t0}"; return 1; } ;;
    esac
    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    case ${k} in
        t0)
            PHYS_BS="${PHYS}" MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${run}" --mode sft \
                --data-prefix "${PRE}" --perm "${p}" --rank 16 --alpha 64 --epochs "${EPOCHS}" --lr 0.0002 \
                --seed "${SEED}" --bs 8 --acc 4 --eval-bs 64 --greedy 1 --gpus "${g}" --end-task 0 \
                > "${jlog}" 2>&1 || rc=$? ;;
        cl)
            PHYS_BS="${PHYS}" bash scripts/qwen/ced/run_cllora.sh --method "${m}" --data-root "data/${PRE}${p}" --num-tasks 10 \
                --rank 16 --alpha 64 --lr 2e-4 --epochs "${EPOCHS}" --batch-size 8 --grad-accum 4 \
                --eval-batch-size 64 --protocol "${PROTO}" --seed "${SEED}" --gpu "${g}" --py "${PY}" \
                > "${jlog}" 2>&1 || rc=$?
            rm -f "${R}/${run}/checkpoint_latest.pt" "${R}/${run}/checkpoint_latest.pt.tmp" ;;
        dist)
            case ${m} in   # run_cre_dist.sh's DistiLLM/AMiD flags
                distillm) kdt=adaptive-srkl; extra="--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000" ;;
                amid)     kdt=adaptive-amid; extra="--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5" ;;
                *)        kdt=${m};          extra="" ;;
            esac
            PHYS_BS="${PHYS}" MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${run}" --mode ce_kd \
                --data-prefix "${PRE}" --perm "${p}" --kd-type "${kdt}" --w-span 0 --kd-ratio 0.9 --skew 0.1 \
                --span-metric cosine --layers "22 25 28" --rank 16 --alpha 64 --epochs "${EPOCHS}" --lr 0.0002 \
                --seed "${SEED}" --bs 8 --acc 4 --eval-bs 64 --greedy 1 --gpus "${g}" --start-task 1 \
                --task0-source-run "${t0}" --extra "${extra}" > "${jlog}" 2>&1 || rc=$? ;;
        ours)   # ours_queue.sh's h2 flags (project_commands_9.sh), from the shared task0
            PHYS_BS="${PHYS}" GRAD_CKPT=1 MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh --run-name "${run}" --mode ce_kd \
                --data-prefix "${PRE}" --perm "${p}" --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 \
                --span-metric cosine --layers "22 25 28" --rank 16 --alpha 64 --epochs "${EPOCHS}" --lr 0.0002 \
                --seed "${SEED}" --bs 2 --acc 16 --eval-bs 64 --pl 1 --replay-boost 5 --greedy 1 --gpus "${g}" \
                --start-task 1 --task0-source-run "${t0}" > "${jlog}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then
        case ${k} in dist|ours) rm -rf "${R}/${run}"/task*/merged ;; esac   # the last task's model, never read again
        log "  done  ${run}"; return 0
    fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog} and ${R}/${run}/task*/train.log"
    return 1
}

# Atomic claim, as in project_commands_6.sh: bash creates the file with O_EXCL. Not `mkdir`: the
# Rust coreutils mkdir of newer Ubuntu (26.04, uutils 0.8) lets two racing callers both succeed.
claim () { ( set -o noclobber; : > "$1" ) 2>/dev/null; }
ready () {  # $1 kind $2 perm: distillation and Ours wait for their perm's shared task0
    case $1 in dist|ours) [ -e "${CLAIMS}/t0_task0_$2.done" ] ;; *) return 0 ;; esac
}
worker () {  # $1 = gpu $2 = slot: take the first unclaimed ready job, then start over from the top
    local g=$1 s=$2 port=$((PORT0 + 10 * $1 + $2)) j k m p left rc n=0
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))   # later slots see the earlier job's memory
    while :; do
        left=0
        for j in "${JOBS[@]}"; do
            IFS='|' read -r k m p <<< "${j}"
            [ -e "${CLAIMS}/${k}_${m}_${p}.done" ] && continue
            left=1
            ready "${k}" "${p}" || continue
            claim "${CLAIMS}/${k}_${m}_${p}" || continue
            rc=0; run_job "${g}" "${port}" "${k}" "${m}" "${p}" || rc=$?
            if [ "${rc}" -eq 2 ]; then   # trained by another process: look again later
                rm -f "${CLAIMS}/${k}_${m}_${p}"; continue
            fi
            [ "${rc}" -eq 0 ] || n=$((n + 1))
            : > "${CLAIMS}/${k}_${m}_${p}.done"
            continue 2
        done
        [ "${left}" = "0" ] && break
        if [ "${DRY}" = "1" ]; then sleep 1; else sleep 60; fi
    done
    log "worker gpu${g}/slot${s} finished, ${n} failed"
}

log "=== project_commands_12: ${DS} sentence-level CRE, steps '${STEPS}', perms '${PERMS}', ${#JOBS[@]} jobs, gpus ${GPUS[*]} x ${SLOTS}, ${EPOCHS} epochs (DRY=${DRY}) ==="
pids=()
for s in $(seq 1 "${SLOTS}"); do
    for g in "${GPUS[@]}"; do worker "${g}" "${s}" & pids+=($!); done
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG} ==="
