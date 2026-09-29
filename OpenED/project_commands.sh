#!/usr/bin/env bash
# Train + test every RAMS baseline (perm0-4, 7 dist + 8 CL-LoRA methods) on this host:
#
#   bash project_commands.sh                    # env + checks, clean once, then trains until done
#   SKIP_INSTALL=1 bash project_commands.sh     # deps already installed
#
# Safe to re-run after a crash: the clean in step 4 runs only once (marker file), finished
# runs are skipped, and a run that died half way resumes from its last finished task.
#
# Knobs (all optional):
#   VENV          venv to activate, e.g. /mnt/local/uvenvs/opened  (default: use the current
#                 environment, whatever `python` already resolves to)
#   PY / ENV_BIN  interpreter and env bin/ for the runners (default: derived from `python`)
#   SKIP_INSTALL  as above
set -euo pipefail
cd "$(dirname "$0")"

step () { echo; echo "=== $* ==="; }
have () { [ -e "$1" ]; }

# ---------------------------------------------------------------- 1. environment
step "1. environment"
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    echo "activated ${VENV}"
else
    echo "no VENV given, using the current environment"
fi
PY=${PY:-$(command -v python || command -v python3)}
[ -x "${PY}" ] || { echo "no python found; set PY or activate an env"; exit 1; }
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
echo "PY=${PY}"
echo "ENV_BIN=${ENV_BIN}"
"${PY}" -V

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1, not touching dependencies"
elif "${PY}" -c "import torch, transformers, peft" 2>/dev/null; then
    echo "torch/transformers/peft already importable, skipping install"
else
    # opened.txt is the uv pin list for hosts that cannot reach GitHub (its en_core_web_sm
    # line was a GitHub wheel URL and has been moved to download.txt as a zip).
    req=opened.txt; [ -f "${req}" ] || req=requirements.txt
    echo "installing from ${req}"
    "${PY}" -m pip install -r "${req}"
fi

# ---------------------------------------------------------------- 2. spaCy model
step "2. spaCy model (en_core_web_sm)"
if have en_core_web_sm/meta.json; then
    echo "already unpacked: $("${PY}" -c "import json;m=json.load(open('en_core_web_sm/meta.json'));print(m['lang']+'_'+m['name'], m['version'])")"
elif have en_core_web_sm.zip; then
    # Python's zipfile, not unzip: the image has no unzip binary and no sudo.
    "${PY}" -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip').extractall('.')"
    rm -rf __MACOSX en_core_web_sm.zip          # the zip was packed on a Mac
    echo "unpacked -> ./en_core_web_sm  (spacy.load(\"en_core_web_sm\") works from this directory)"
else
    echo "no en_core_web_sm.zip here -- download.txt fetches it; skipping (nothing in this repo imports spaCy)"
fi

# ---------------------------------------------------------------- 3. data
step "3. data"
missing=0
for f in data/rams_b10_perm0/streams.json data/rams_b10_perm0/0/train.jsonl \
         processed_data/rams_b10_perm0/0/qwen/train_0.idx; do
    if have "${f}"; then
        printf '  %-44s ok\n' "${f}"
    else
        printf '  %-44s MISSING\n' "${f}"
        missing=1
    fi
done
for f in data/tacred_perm0/streams.json data/fewrel_perm0/streams.json data/tacred_groups/groups.json; do
    have "${f}" && printf '  %-44s ok\n' "${f}" || printf '  %-44s absent (CRE, fine if this host only runs CED)\n' "${f}"
done
[ "${missing}" = "0" ] || {
    echo "CED splits are missing -- fetch them with download.txt (they land in these exact"
    echo "directory names, nothing to rename), then re-run this script."
    exit 1
}

# ---------------------------------------------------------------- 4. clean what gets retrained
# Hard-coded from the 2026-09-28 collect. Edit these lists by hand.
#
# KEPT (perm0, finished at batch 2x16 = 32, never deleted, skipped by the pool below):
#   dist_shared_task0_perm0  dist_rkl_perm0  dist_distillm_perm0
#   cllora_{inclora,olora,tree,inflora,epi,migu,gainlora_o,gainlora_inf}_perm0
#
# RETRAINED at batch 32x1 (deleted here, then trained again so their logs land in logs/):
#   perm0    dist kd sfkl srkl (died at task1), csd amid (never started)
#   perm1-4  everything, dist and CL-LoRA, finished or not
#
# Runs once: the marker file stops a re-run after a crash from deleting what trained since.
R=results/qwen3/ced
CLEANED=${R}/.cleaned_2026-09-28
step "4. clean"
if [ -f "${CLEANED}" ]; then
    echo "already cleaned on $(cat "${CLEANED}"), deleting nothing"
else
    if pgrep -f "scripts/qwen/ced/(dist_queue|run_all_cllora|run_ced_v2|run_cllora)\.sh" >/dev/null; then
        echo "training processes are still running, kill them first:"
        pgrep -af "scripts/qwen/ced/(dist_queue|run_all_cllora|run_ced_v2|run_cllora)\.sh"
        exit 1
    fi
    shopt -s nullglob
    for d in "${R}"/dist_{kd,sfkl,srkl,csd,amid}_perm0_rams_v2_s42 "${R}"/*_perm[1-4]_rams_v2_s42; do
        [ -e "${d}" ] || continue
        echo "  rm ${d}"
        rm -rf "${d}"
    done
    shopt -u nullglob
    mkdir -p "${R}"
    date -Iseconds > "${CLEANED}"
fi

# ---------------------------------------------------------------- 5. train, one run per GPU
# Every job below is one method on one perm, on one GPU. When a GPU frees up it takes the
# first job in JOBS that can start: a dist job waits for the shared task0 of its perm.
# Up to 8 torchruns at once, so each GPU gets its own master port (29500 + gpu) instead of
# run_ced_v2.sh's random one out of 90, which collides about 1 time in 4 at 8 jobs.
GPUS=(0 1 2 3 4 5 6 7)
DIST_ALL="kd rkl sfkl srkl csd distillm amid"
CLLORA_ALL="inclora olora tree inflora epi migu gainlora_o gainlora_inf"
# All 5 perms, KEPT perm0 runs included: pick() skips anything with a .complete marker, so a
# kept run that really finished costs nothing, and one that never did gets trained here
# instead of being assumed done. Status 2026-09-29: all CL-LoRA perm1-4 done, task0 perm1
# done; every dist job and task0 perm2-4 failed on the shared rendezvous (fixed below).
JOBS=()                                                  # <kind>:<method>:<perm>
for p in 0 1 2 3 4; do JOBS+=("task0:shared:${p}"); done # first, everything dist waits on them
for p in 0 1 2 3 4; do
    for m in ${DIST_ALL}; do JOBS+=("dist:${m}:${p}"); done
    for m in ${CLLORA_ALL}; do JOBS+=("cllora:${m}:${p}"); done
done

step "5. train ${#JOBS[@]} jobs on gpus ${GPUS[*]}"
bash run.sh rams "0 1 2 3 4" 0 0 prep                   # tokenize what is missing, trains nothing
# Same defaults run.sh gives its queues: local model copy -> offline, or the hub hangs.
if [ -f models/Qwen3-0.6B/config.json ]; then
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi
# The pod exports PET_RDZV_BACKEND=c10d / PET_RDZV_ENDPOINT=<pod>-worker-0:23456 / PET_RDZV_ID,
# which torchrun reads as flag defaults. c10d ignores --master_port, so every torchrun joined the
# same rendezvous: the first one ran, the rest crashed when it exited (RendezvousConnectionError).
# Dropped here, not in run_ced_v2.sh: that script is fingerprinted into every run manifest, and
# editing it would break --resume of every partial dist run.
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
POOL_LOG=logs/rams_pool.log
mkdir -p logs
echo "progress: ${POOL_LOG}   full logs: logs/rams_{dist,cllora}_<method>_perm<p>_*.log"
log () { echo "[pool $(date '+%F %T')] $*" | tee -a "${POOL_LOG}"; }

run_dir () {  # $1=kind $2=method $3=perm -> results dir of that run
    case $1 in
        task0)  echo "${R}/dist_shared_task0_perm$3_rams_v2_s42" ;;
        dist)   echo "${R}/dist_$2_perm$3_rams_v2_s42" ;;
        cllora) echo "${R}/cllora_$2_perm$3_rams_v2_s42" ;;
    esac
}

BATCH=32   # the micro batch dist_queue.sh and run_all_cllora.sh train with; keep in sync
# A partial run from an earlier batch size (the 128 and 64 runs that OOM'd) can never resume: its
# manifest pins micro_batch, so --resume dies on "manifest mismatch". Start those over.
drop_stale () {  # $1=job $2=run dir
    local f="$2/run_manifest.json" b=""
    [ -e "$2" ] && [ ! -f "$2/.complete" ] || return 0
    [ -f "${f}" ] && b=$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("micro_batch"))' "${f}")
    [ "${b}" = "${BATCH}" ] && return 0
    log "reset  $1 (partial run at micro_batch=${b:-?}, now ${BATCH})"
    rm -rf "$2"
}

launch () {  # $1=job $2=gpu -> starts it in the background, output appended to POOL_LOG
    local kind m p dir resume=0
    IFS=: read -r kind m p <<< "$1"
    dir=$(run_dir "${kind}" "${m}" "${p}")
    drop_stale "$1" "${dir}"
    case ${kind} in
        task0)
            # one task only, so a half-trained one restarts: dist_queue.sh refuses to reuse it
            rm -rf "${dir}"
            PERM=${p} GPU=$2 PROTOCOL=rams_v2 DATA_PREFIX=rams_b10_perm DIST_METHODS="" \
                MASTER_PORT=$((29500 + $2)) bash scripts/qwen/ced/dist_queue.sh ;;
        dist)
            # RESUME=1 only acts when the run dir already exists (dist_queue.sh checks)
            PERM=${p} GPU=$2 PROTOCOL=rams_v2 DATA_PREFIX=rams_b10_perm DIST_METHODS=${m} \
                RESUME=1 MASTER_PORT=$((29500 + $2)) bash scripts/qwen/ced/dist_queue.sh ;;
        cllora)
            # --resume needs the per-task checkpoint; a run that died inside task0 has none
            # and cannot restart fresh either (run_cllora.sh refuses to overwrite it)
            if [ -f "${dir}/checkpoint_latest.pt" ]; then resume=1; else rm -rf "${dir}"; fi
            RESUME=${resume} DATA_ROOT=data/rams_b10_perm${p} PROTOCOL=rams_v2 \
                bash scripts/qwen/ced/run_all_cllora.sh "$2" "${m}" ;;
    esac >> "${POOL_LOG}" 2>&1 &
}

# task0 state per perm: done | pending | running | failed
T0=()                                                    # indexed by perm
for p in 0 1 2 3 4; do
    [ -f "$(run_dir task0 - "${p}")/.complete" ] && T0[${p}]=done || T0[${p}]=pending
done

pick () {  # sets JOB to the first startable job and drops it from PENDING; 1 if none
    local i job kind m p
    for i in "${!PENDING[@]}"; do
        job=${PENDING[i]}
        IFS=: read -r kind m p <<< "${job}"
        if [ -f "$(run_dir "${kind}" "${m}" "${p}")/.complete" ]; then
            log "skip   ${job} (already complete)"
            unset 'PENDING[i]'; continue
        fi
        if [ "${kind}" = "dist" ]; then
            case ${T0[${p}]} in
                pending|running) continue ;;
                failed) log "FAILED ${job} (task0 of perm${p} failed)"; n_fail=$((n_fail + 1))
                        unset 'PENDING[i]'; continue ;;
            esac
        fi
        JOB=${job}; unset 'PENDING[i]'
        [ "${kind}" = "task0" ] && T0[${p}]=running
        return 0
    done
    return 1
}

PENDING=("${JOBS[@]}")
declare -a SLOT_PID SLOT_JOB
n_fail=0
while :; do
    for i in "${!GPUS[@]}"; do
        pid=${SLOT_PID[i]:-}
        if [ -n "${pid}" ]; then
            kill -0 "${pid}" 2>/dev/null && continue
            rc=0; wait "${pid}" || rc=$?
            job=${SLOT_JOB[i]}; IFS=: read -r kind m p <<< "${job}"
            if [ "${rc}" -eq 0 ] && [ -f "$(run_dir "${kind}" "${m}" "${p}")/.complete" ]; then
                log "done   ${job} (gpu${GPUS[i]})"
                [ "${kind}" = "task0" ] && T0[${p}]=done
            else
                log "FAILED ${job} (gpu${GPUS[i]}, exit ${rc}), see logs/rams_*${m}*perm${p}_*.log"
                n_fail=$((n_fail + 1))
                [ "${kind}" = "task0" ] && T0[${p}]=failed
            fi
            SLOT_PID[i]=""
        fi
        pick || continue
        launch "${JOB}" "${GPUS[i]}"
        SLOT_PID[i]=$!; SLOT_JOB[i]=${JOB}
        log "start  ${JOB} (gpu${GPUS[i]})"
    done
    busy=0
    for pid in ${SLOT_PID[@]+"${SLOT_PID[@]}"}; do [ -n "${pid}" ] && busy=1; done
    if [ "${busy}" = "0" ]; then
        # nothing running and nothing startable: whatever is left can never start
        for job in ${PENDING[@]+"${PENDING[@]}"}; do log "FAILED ${job} (never startable)"; n_fail=$((n_fail + 1)); done
        break
    fi
    sleep 20
done

step "6. done"
if [ "${n_fail}" -gt 0 ]; then
    echo "${n_fail} jobs failed: grep FAILED ${POOL_LOG}"
    exit 1
fi
echo "all jobs finished. Collect the F1 files: bash gather_logs.sh"
