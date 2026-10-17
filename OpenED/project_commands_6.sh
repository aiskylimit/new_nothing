#!/usr/bin/env bash
# SDFT baseline (Shenfeld et al., 2026; code in ../../Self-Distillation), on every dataset and perm.
#
#   bash project_commands_6.sh                          # SDFT + CE, gpus 4 and 5, all datasets
#   SD_CE=0 bash project_commands_6.sh                  # pure SDFT, the first version of this file
#   DRY=1 bash project_commands_6.sh                    # print the plan, train nothing
#   DATASETS="ace geneva" PERMS="1 2" bash project_commands_6.sh
#
# SD_CE=1 (default since 2026-10-07): pure SDFT collapsed (ACE Last 21.1, GENEVA 3.7, MAVEN 0):
# the 0.6B teacher copies the gold answer from its prompt too weakly to teach the new types
# (new-type recall after task1: ACE 0.30, GENEVA 0.08), and with nothing else in the loss the
# teacher's entropy grows every step. This variant adds cross-entropy on every row's gold target
# and changes nothing else: loss = CE + 1.0 * SD on every micro-batch (--kd-ratio 0, so the CE
# weight does not drop on micro-batches with replay rows). Run names dist_sdftce_*, so the pure
# SDFT runs (dist_sdft_*) are kept and not skipped as done.
#
# Self-distillation only, as in the SDFT release (main.py + distil_trainer.py), through the SD
# path of ced_finetune.py that was ported from it:
#   * teacher = EMA of the student, updated every optimizer step, alpha 0.01 (--sd-mu 0.99)
#   * teacher prompt = "<prompt>\n\nThis is an example for a response to the question:\n<gold>
#     \n\nNow answer with a response of your own." (tools/ced_sd_prompts.py; the gold answer is
#     the task's own annotation, no pseudo-labels)
#   * one sampled response per prompt, temperature 1.0, top_p 1.0; forward KL over the vocab
#   * the first 3 response tokens are left out of the loss (num_loss_tokens_to_skip = 3)
#   * the loss is the SD term alone: no CE, no KD from the previous model (--ced-sd-only),
#     no span loss, no omission-mask, no replay oversampling
# Everything else follows the protocol of the other baselines: <ds>_b10_perm data (buffer 10 per
# type; <ds>_perm for TACRED/FewRel), the shared plain-CE task0 checkpoint of the distillation
# group, LoRA r16/a64, 5 epochs, lr 2e-4, effective batch 32 (one micro-batch of 32).
#
# Speed (none of these changes what is trained or the number the tables read):
#   * micro-batch 32 x 1: one generate() call of 32 samples per update instead of 4 of 8; a
#     0.6B model generating in small batches leaves the GPU mostly idle
#   * evaluation only after the last update of each task (--eval-interval -2): the per-epoch
#     dev+test generation was most of the wall time, and SDFT never reads it during training
#   * no SD probe (a diagnostic only); eval batch stays at the runner default (32), as for every other run
#   * SLOTS jobs per GPU, each started at once (NEED_MB=80000 makes it wait for that much free
#     memory first): default 1 with CE
#     (its extra forward at micro-batch 32 needs more memory), 2 for pure SDFT
#
# Jobs are (dataset, perm), smallest datasets first; each slot takes the next unclaimed job.
# Run names: dist_sdftce_perm<p>_<ds>_v2_s42 (dist_sdft_* with SD_CE=0),
# task0 dist_shared_task0_perm<p>_<ds>_v2_s42.
# Safe to re-run: finished runs are skipped; a crashed partial one is moved to
# results/qwen3/ced/_failed/ (never deleted) and retrained.
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
SD_CE=${SD_CE:-1}            # 1 = SDFT + CE, 0 = pure SDFT
GPUS=(${GPUS:-4 5})          # 0 1 until 2026-10-07 (pure SDFT)
if [ "${SD_CE}" = "1" ]; then SLOTS=${SLOTS:-1}; else SLOTS=${SLOTS:-2}; fi   # concurrent jobs per GPU
NEED_MB=${NEED_MB:-0}        # free GPU MiB a job waits for before it starts; 0 = start at once
DATASETS=${DATASETS:-"ace geneva tacred rams fewrel maven"}   # rows to train, small -> large
PERMS=${PERMS:-"0 1 2 3 4"}
SEED=${SEED:-42}
R=results/qwen3/ced
if [ "${SD_CE}" = "1" ]; then
    NAME=sdftce
    LOSS_ARGS=(--kd-ratio 0)          # CE + SD; full CE weight on every micro-batch
    SD_ONLY=""
else
    NAME=sdft
    LOSS_ARGS=()
    SD_ONLY="--ced-sd-only "          # the SD term is the whole update
fi
CLAIMS=logs/${NAME}_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/${NAME}_baseline.log
log () { echo "[${NAME} $(date '+%F %T')] $*" | tee -a "${LOG}"; }

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

live () { pgrep -f -- "run-name $1( |$)" > /dev/null; }
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
prefix_of () { case $1 in tacred|fewrel) echo "$1_perm" ;; *) echo "$1_b10_perm" ;; esac; }

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

job () {  # $1 = gpu $2 = master port $3 = dataset $4 = perm. 1 = failed
    local g=$1 port=$2 ds=$3 p=$4 rc=0
    local pre; pre=$(prefix_of "${ds}")
    local proto="${ds}_v2"
    local shared="dist_shared_task0_perm${p}_${proto}_s${SEED}"
    local run="dist_${NAME}_perm${p}_${proto}_s${SEED}"
    local logp="logs/${ds}_dist_${NAME}_perm${p}"
    [ -s "data/${pre}${p}/streams.json" ] || { log "  missing data/${pre}${p}, skip ${run}"; return 1; }
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    if live "${run}"; then log "  skip ${run} (running elsewhere)"; return 0; fi
    ensure_tokenized "${pre}" "${p}" || { log "  FAILED tokenising ${pre}${p}"; return 1; }

    # shared plain-CE task0, the one every distillation baseline of this perm starts from
    while live "${shared}"; do
        log "  ${shared} is training elsewhere, waiting"; [ "${DRY}" = "1" ] && break; sleep 300
    done
    if [ ! -f "${R}/${shared}/.complete" ]; then
        [ -e "${R}/${shared}" ] && park "${shared}"
        wait_gpu "${g}"
        log "  start ${shared} on gpu${g}"
        if [ "${DRY}" != "1" ]; then
            PERM="${p}" GPU="${g}" PROTOCOL="${proto}" SEED="${SEED}" DATA_PREFIX="${pre}" DIST_METHODS="" \
            MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh >> "${LOG}" 2>&1 || rc=$?
            [ "${rc}" -eq 0 ] && [ -f "${R}/${shared}/.complete" ] \
                || { log "  FAILED ${shared} (exit ${rc}); skip ${run}"; return 1; }
        fi
    fi

    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    MASTER_PORT="${port}" bash scripts/qwen/ced/run_ced_v2.sh \
        --run-name "${run}" --mode ce_kd --data-prefix "${pre}" --perm "${p}" \
        --kd-type no --w-span 0 --pl 0 ${LOSS_ARGS[@]+"${LOSS_ARGS[@]}"} \
        --sd 1 --w-sd 1.0 --sd-mu 0.99 --sd-temp 1.0 --sd-top-p 1.0 --sd-div fkl \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 32 --acc 1 \
        --greedy 1 --gpus "${g}" --start-task 1 --task0-source-run "${shared}" \
        --extra "${SD_ONLY}--ced-sd-skip-tokens 3 --ced-sd-probe 0 --eval-interval -2" \
        > "${logp}_steps.log" 2>&1 || rc=$?
    : > "${logp}_results.log"   # per-task log.txt dump, as dist_queue.sh writes
    for f in $(find "${R}/${run}" -name log.txt 2>/dev/null | sort -V); do
        echo "===== ${f#${R}/${run}/} =====" >> "${logp}_results.log"; cat "${f}" >> "${logp}_results.log"
    done
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${logp}_steps.log"
    return 1
}

# Atomic claim: bash creates the file with O_EXCL. Not `mkdir`: the Rust coreutils mkdir of newer
# Ubuntu (26.04, uutils 0.8) lets two racing callers both succeed, so two workers could train
# the same run at once.
claim () { ( set -o noclobber; : > "$1" ) 2>/dev/null; }

worker () {  # $1 = gpu $2 = slot: take the next unclaimed (dataset, perm)
    local g=$1 s=$2 ds p n=0
    local port=$((29900 + 10 * g + s))
    # later slots start a little later, so the GPU memory check sees the earlier job's usage
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))
    for ds in ${DATASETS}; do
        for p in ${PERMS}; do
            claim "${CLAIMS}/${ds}_${p}" || continue
            job "${g}" "${port}" "${ds}" "${p}" || n=$((n + 1))
        done
    done
    log "worker gpu${g}/slot${s} finished, ${n} failed"
}

log "=== SDFT baseline (${NAME}): datasets '${DATASETS}', perms '${PERMS}', gpus ${GPUS[*]} x ${SLOTS} slots (DRY=${DRY}) ==="
pids=()
for s in $(seq 1 "${SLOTS}"); do
    for g in "${GPUS[@]}"; do worker "${g}" "${s}" & pids+=($!); done
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG}; results in ${R}/dist_${NAME}_perm*_*_v2_s${SEED}/ ==="
