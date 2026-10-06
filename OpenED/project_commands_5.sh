#!/usr/bin/env bash
# "Our distillation on CL-LoRA methods" table: TreeLoRA / IncLoRA / O-LoRA + Ours on ACE, 5 perms.
#
#   bash project_commands_5.sh                  # gpu0: tree, gpu1: inclora, gpu6: olora
#   DRY=1 bash project_commands_5.sh            # print the plan, train nothing
#   PERMS="0" PLAN="0:tree" bash project_commands_5.sh
#
# "+ Ours" = the CL-LoRA method unchanged (its own adapters and regulariser) plus our
# distillation, the f12_pl recipe that is the Ours row of every table (cl_lora/ours.py):
#   before each task t > 0: pseudo-labels from the previous model (conflict dedup + lexicon,
#   no confidence filter), replay rows oversampled x5;
#   loss on batches with replay rows: 0.1 CE + 0.9 (SFKL skew 0.1 + 2.0 span loss, layers
#   22 25 28, cosine); teacher = the model at the start of the task. Task 0 is the plain method.
#
# One worker per GPU, one method each (PLAN = "gpu:method ..."), one run at a time per GPU,
# perms in order. The three methods cost about the same, so the workers finish together.
# Batch 8 x 4 (effective 32): the token KD holds batch x seq x vocab fp32 tensors.
# Safe to re-run: a finished run is skipped; a crashed partial one is moved to
# results/qwen3/ced/_failed/ (never deleted) and retrained.
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
PLAN=${PLAN:-"0:tree 1:inclora 6:olora"}
PERMS=${PERMS:-"0 1 2 3 4"}
SEED=${SEED:-42}
R=results/qwen3/ced
mkdir -p logs "${R}/_failed"
LOG=logs/ace_cllora_ours.log
log () { echo "[ours5 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---------------------------------------------------------------- environment (as project_commands.sh)
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then set +u; source "${VENV}/bin/activate"; set -u; fi
PY=${PY:-$(command -v python || command -v python3)}
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
if [ -f models/Qwen3-0.6B/config.json ]; then
    MODEL_PATH=models/Qwen3-0.6B
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

run_one () {  # $1 = gpu $2 = method $3 = perm. 1 = failed
    local g=$1 m=$2 p=$3 rc=0
    local data="data/ace_b10_perm${p}" run="cllora_$2_ours_perm$3_ace_v2_s${SEED}"
    local tlog="logs/ace_cllora_ours_$2_perm$3_train.log"
    [ -s "${data}/streams.json" ] || { log "  missing ${data}, skip ${run}"; return 1; }
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if pgrep -f -- "--save ${R}/${run}( |$)" > /dev/null; then log "  skip ${run} (running elsewhere)"; return 0; fi
    if [ -e "${R}/${run}" ]; then
        local dst="${R}/_failed/${run}_$(date +%Y%m%d_%H%M)"
        log "  moving partial ${R}/${run} -> ${dst}"
        [ "${DRY}" = "1" ] || mv "${R}/${run}" "${dst}"
    fi
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    CUDA_VISIBLE_DEVICES="${g}" PYTHONPATH=. "${PY}" cl_lora/engine.py \
        --cl-method "${m}" --ours --data-root "${data}" --num-tasks 5 \
        --rank 16 --alpha 64 --lr 2e-4 --epochs 5 \
        --batch-size 8 --grad-accum 4 --eval-batch-size 16 \
        --seed "${SEED}" --model-path "${MODEL_PATH}" --save "${R}/${run}" \
        > "${tlog}" 2>&1 || rc=$?
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then
        log "  done  ${run}: $(grep -o 'CL-LORA DONE.*' "${tlog}" | tail -1)"
        return 0
    fi
    log "  FAILED ${run} (exit ${rc}), see ${tlog}"
    return 1
}
worker () {  # $1 = gpu $2 = method
    local p n=0
    for p in ${PERMS}; do run_one "$1" "$2" "${p}" || n=$((n + 1)); done
    log "worker gpu$1 ($2) finished, ${n} failed"
}

log "=== +Ours on CL-LoRA, ACE perms '${PERMS}', plan '${PLAN}' (DRY=${DRY}) ==="
pids=()
for entry in ${PLAN}; do
    worker "${entry%%:*}" "${entry#*:}" & pids+=($!)
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG}; results in ${R}/cllora_*_ours_perm*_ace_v2_s${SEED}/cl_results.json ==="
