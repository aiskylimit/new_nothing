#!/usr/bin/env bash
# Train + test every RAMS baseline (perm0-4, 7 dist + 8 CL-LoRA methods) on this host:
#
#   bash project_commands.sh                    # env + checks, then trains until done
#   SKIP_INSTALL=1 bash project_commands.sh     # deps already installed
#
# Everything here is idempotent: re-running skips what is already in place, and the runners
# skip every method+perm that already has its completion marker, so a re-run after a crash
# only trains what is missing.
#
# Knobs (all optional):
#   VENV          venv to activate, e.g. /mnt/local/uvenvs/opened  (default: use the current
#                 environment, whatever `python` already resolves to)
#   PY / ENV_BIN  interpreter and env bin/ for the runners (default: derived from `python`)
#   GPU_DIST_ALL / GPU_CLLORA_ALL   GPUs per queue (default 0,1,2,3 / 4,5,6,7, see run.sh)
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

# ---------------------------------------------------------------- 4. full sweep
step "4. sweep: rams perm0-4, dist (gpu ${GPU_DIST_ALL:-0,1,2,3}) + CL-LoRA (gpu ${GPU_CLLORA_ALL:-4,5,6,7})"
# results/ is gitignored, so a fresh host has no completion markers and nothing is skipped.
# MISSING_PLAN keeps it to RAMS; a bare `bash run.sh` would also retrain MAVEN.
# FOREGROUND=1: run.sh trains in this process instead of detaching, so this script returns
# only when every run is done. The two queue logs are streamed here as they fill.
tail -n 0 -F logs_ced_dist_rams.log logs_ced_cllora_rams.log 2>/dev/null &
tail_pid=$!
FOREGROUND=1 MISSING_PLAN="rams:0 1 2 3 4:both" bash run.sh
kill "${tail_pid}" 2>/dev/null || true

step "5. done"
if grep -h "FAILED" logs_ced_dist_rams.log logs_ced_cllora_rams.log; then
    echo "some runs failed, see the lines above"
    exit 1
fi
echo "sweep finished. Collect: python tools/ced_collect.py --host-label <label> [--upload]"
