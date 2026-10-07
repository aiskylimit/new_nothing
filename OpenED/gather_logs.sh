#!/bin/bash
# Collect only the files needed to read per-task F1 of every run into one folder, plus the end of
# each training log, and pack the folder into a .tar.gz next to it. No checkpoints, no model outputs.
#
#   bash gather_logs.sh                  # -> collected_logs/<host>_<date>/ and collected_logs/<host>_<date>.tar.gz
#   bash gather_logs.sh a100_run1        # -> collected_logs/a100_run1/ and collected_logs/a100_run1.tar.gz
#   WITH_LOGS=0 bash gather_logs.sh      # results/ files only, as before
#
# Copied from results/qwen3/ced/<run>/, repo layout kept:
#   task*/**/log.txt     dist runs: one "dev | ..." / "test | ..." line per eval, F1 inside
#   cl_results.json      CL-LoRA runs: {"task<t>": {"trigger": {"f1": ...}}, ...}
#   run_config.txt, run_manifest.json, .complete   which config, how far it got, finished or not
# With WITH_LOGS=1 (the default), the last LOG_LINES lines of every logs/*.log / *.out, every
# ./logs_<run>.log (ours_queue.sh's per-run stdout) and the newest task*/train.log of each
# unfinished run: where a crashed run leaves its traceback or OOM. Progress-bar updates (\r) count
# as lines, so one long tqdm line cannot hide the end. *_results.log is skipped: it repeats the
# log.txt files above.
# Download the .tar.gz: about 8x smaller than the folder (~3 MB against ~23 MB on 07/10).
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${1:-$(hostname -s)_$(date +%Y%m%d_%H%M)}
WITH_LOGS=${WITH_LOGS:-1}
LOG_LINES=${LOG_LINES:-300}
OUT=collected_logs/${LABEL}
[ -d results ] || { echo "no results/ here"; exit 1; }
[ ! -e "${OUT}" ] || { echo "${OUT} already exists, pick another label"; exit 1; }
mkdir -p "${OUT}"

n=0
while IFS= read -r -d '' f; do
    mkdir -p "${OUT}/$(dirname "$f")"
    cp -p "$f" "${OUT}/$f"
    n=$((n + 1))
done < <(find results -path '*/merged' -prune -o -type f \( -name log.txt -o -name cl_results.json \
    -o -name run_config.txt -o -name run_manifest.json -o -name .complete \) -print0)

m=0
tail_into () {  # $1 = file: its last LOG_LINES lines, \r split, at the same path under OUT
    mkdir -p "${OUT}/$(dirname "$1")"
    tr '\r' '\n' < "$1" | tail -n "${LOG_LINES}" > "${OUT}/$1"
    m=$((m + 1))
}
if [ "${WITH_LOGS}" = "1" ]; then
    if [ -d logs ]; then
        while IFS= read -r -d '' f; do tail_into "$f"; done \
            < <(find logs -type f \( -name '*.log' -o -name '*.out' \) ! -name '*_results.log' -print0)
    fi
    # ours_queue.sh writes each run's stdout next to logs/, as logs_<run>.log
    while IFS= read -r -d '' f; do tail_into "$f"; done \
        < <(find . -maxdepth 1 -type f -name 'logs_*.log' -printf '%P\0')
    # an unfinished run: its newest train.log, which is where it crashed or where it is now
    for d in results/qwen3/ced/*/; do
        if [ "$(basename "${d}")" = "_failed" ] || [ -f "${d}.complete" ]; then continue; fi
        f=$(find "${d}" -name train.log -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "${f}" ]; then tail_into "${f}"; fi
    done
fi

echo "files:  ${n} from results/, ${m} log tails"
echo "folder: ${OUT} ($(du -sh "${OUT}" | cut -f1))"

# the image may have no tar binary (see project_commands.sh), so fall back to Python's tarfile
TGZ=${OUT}.tar.gz
if command -v tar > /dev/null; then
    tar czf "${TGZ}" -C collected_logs "${LABEL}"
elif command -v python3 > /dev/null; then
    python3 -c "import sys, tarfile; tarfile.open(sys.argv[1], 'w:gz').add(sys.argv[2], arcname=sys.argv[3])" \
        "${TGZ}" "${OUT}" "${LABEL}"
else
    echo "no tar and no python3: download the folder instead"; exit 0
fi
echo "packed: ${TGZ} ($(du -sh "${TGZ}" | cut -f1))"
size=$(du -k "${TGZ}" | cut -f1)
[ "${size}" -le 25600 ] || echo "WARNING: ${TGZ} is over 25 MB; try LOG_LINES=100 or WITH_LOGS=0"
