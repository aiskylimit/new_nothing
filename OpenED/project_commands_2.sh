#!/usr/bin/env bash
# The runs the paper tables still need that project_commands.sh does not cover: baselines.
#   geneva       all 15 baselines (7 distillation + 8 CL-LoRA) on GENEVA, perms 0-4. The CEE
#                table and the argument table have a GENEVA block and nothing has run there yet
#   tacred_gap   TACRED GainLoRA(InfLoRA) perm4, the only TACRED cell still on 4 perms
#
#   bash project_commands_2.sh                       # both steps, in order
#   STEPS="geneva" bash project_commands_2.sh        # one step
#   DRY=1 bash project_commands_2.sh                 # print what would run
#
# Run it after project_commands.sh, not next to it: both default to GPUs 4-7, and the runners
# allow one queue per GPU. Every runner skips a run that already has its .complete marker.
#
# Knobs (all optional):
#   VENV / PY / ENV_BIN / SKIP_INSTALL   as in project_commands.sh
#   GPU_DIST      GPUs for the distillation queue, comma list   (default 4,5)
#   GPU_CLLORA    GPUs for the CL-LoRA queue, comma list        (default 6,7; must not overlap)
#   PERMS         GENEVA perms                                  (default "0 1 2 3 4")
#   STEPS         default "geneva tacred_gap"
set -euo pipefail
cd "$(dirname "$0")"

step () { echo; echo "=== $* ==="; }
have () { [ -e "$1" ]; }
DRY=${DRY:-0}
STEPS=${STEPS:-"geneva tacred_gap"}
PERMS=${PERMS:-"0 1 2 3 4"}
GPU_DIST=${GPU_DIST:-4,5}
GPU_CLLORA=${GPU_CLLORA:-6,7}
wants () { [[ " ${STEPS} " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 1. environment
step "1. environment"
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    set +u; source "${VENV}/bin/activate"; set -u   # activate scripts read unset vars
    echo "activated ${VENV}"
else
    echo "no VENV given, using the current environment"
fi
PY=${PY:-$(command -v python || command -v python3)}
[ -x "${PY}" ] || { echo "no python found; set PY or activate an env"; exit 1; }
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
# the baseline runners default to conda envs on the A40 hosts (envs/mta, envs/nuquant)
export PY ENV_BIN
echo "PY=${PY}"; echo "ENV_BIN=${ENV_BIN}"; "${PY}" -V

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1, not touching dependencies"
elif "${PY}" -c "import torch, transformers, peft" 2>/dev/null; then
    echo "torch/transformers/peft already importable, skipping install"
else
    req=opened.txt; [ -f "${req}" ] || req=requirements.txt
    echo "installing from ${req}"
    "${PY}" -m pip install -r "${req}"
fi

# same pod fix as project_commands.sh: torchrun would read the pod's rendezvous settings
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
if [ -f .env ] && [ -z "${HF_TOKEN:-}" ]; then
    set -a; . ./.env; set +a
    echo "read HF_TOKEN from .env"
fi
if [ -f models/Qwen3-0.6B/config.json ]; then
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi
export DISK_PATH=${DISK_PATH:-.}   # the runners' free-space guard; /mnt only exists on the A40 hosts

# ---------------------------------------------------------------- 2. data
step "2. data"
for tgz in *_all.tar.gz; do
    have "${tgz}" || continue
    echo "  unpacking ${tgz}"
    if [ "${DRY}" != "1" ]; then
        "${PY}" -c "import sys, tarfile; tarfile.open(sys.argv[1]).extractall('.', filter='data')" "${tgz}"
        rm -f "${tgz}"
    fi
done
need=()
wants geneva && for p in ${PERMS}; do need+=("data/geneva_b10_perm${p}/streams.json"); done
wants tacred_gap && need+=("data/tacred_perm4/streams.json")
for f in ${need[@]+"${need[@]}"}; do
    have "${f}" && { printf '  %-40s ok\n' "$(dirname "${f}")"; continue; }
    echo "  $(dirname "${f}") MISSING: fetch the archives in download.txt first"
    exit 1
done
mkdir -p logs

# ---------------------------------------------------------------- 3. GENEVA baselines
n_fail=0
if wants geneva; then
    step "3. GENEVA: 15 baselines, perms '${PERMS}' (dist on ${GPU_DIST}, CL-LoRA on ${GPU_CLLORA})"
    # run.sh no-argument mode with a one-entry plan: tokenizes, runs both queues side by side
    # and waits for both. FOREGROUND=1 keeps it attached so this script waits too.
    # GENEVA's run names carry the tag geneva_v2 (run.sh protocol_of), so they never meet ACE's.
    echo "  logs: logs/run_all.log, logs/geneva_{dist,cllora}_queue.log, logs/geneva_*_perm<p>_*.log"
    if [ "${DRY}" = "1" ]; then
        echo "  DRY=1: MISSING_PLAN=\"geneva:${PERMS}:both\" FOREGROUND=1 bash run.sh"
    else
        MISSING_PLAN="geneva:${PERMS}:both" FOREGROUND=1 \
            GPU_DIST_ALL="${GPU_DIST}" GPU_CLLORA_ALL="${GPU_CLLORA}" bash run.sh || n_fail=$((n_fail + 1))
        # the queues report failures in their logs rather than in run.sh's exit code
        if grep -h "FAILED" logs/run_all.log logs/geneva_dist_queue.log logs/geneva_cllora_queue.log 2>/dev/null; then
            n_fail=$((n_fail + 1))
        fi
    fi
fi

# ---------------------------------------------------------------- 4. TACRED gap
if wants tacred_gap; then
    gpu=${GPU_CLLORA%%,*}
    step "4. TACRED: GainLoRA(InfLoRA) perm4 on gpu${gpu}"
    # METHODS narrows the CRE CL-LoRA runner to the one missing method; the other seven perm4
    # runs live on the host that ran them and must not be retrained here
    if [ "${DRY}" = "1" ]; then
        echo "  DRY=1: METHODS=gainlora_inf bash scripts/qwen/cre/run_cre_cllora.sh tacred 4 ${gpu}"
    else
        METHODS=gainlora_inf bash scripts/qwen/cre/run_cre_cllora.sh tacred 4 "${gpu}" \
            > logs/tacred_cllora_gainlora_inf_perm4_queue.log 2>&1 || n_fail=$((n_fail + 1))
        echo "  log: logs/tacred_cllora_gainlora_inf_perm4_queue.log"
    fi
fi

# ---------------------------------------------------------------- 5. collect
step "5. collect F1 files"
if [ "${DRY}" = "1" ]; then
    echo "DRY=1: would run gather_logs.sh baselines_$(date +%Y%m%d_%H%M)"
else
    bash gather_logs.sh "baselines_$(date +%Y%m%d_%H%M)" || echo "gather_logs.sh failed, run it by hand"
fi

step "6. done"
if [ "${n_fail}" -gt 0 ]; then
    echo "${n_fail} step(s) reported failures, see the logs above"
    exit 1
fi
echo "all steps finished, F1 files are under collected_logs/"
