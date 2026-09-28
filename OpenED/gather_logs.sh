#!/bin/bash
# Collect every log and result file of this repo into one folder, without model weights.
#
#   bash gather_logs.sh                  # -> collected_logs/<host>_<date>/
#   bash gather_logs.sh a100_run1        # -> collected_logs/a100_run1/
#   MAX_MB=200 bash gather_logs.sh       # raise the per-file size cap (default 50 MB)
#
# What goes in, repo layout kept:
#   logs_*.log        queue/runner stdout in the repo root
#   /tmp/tok_*.log    tokenization logs written by run.sh's check_data   -> tmp/
#   results/**        every run dir: run_manifest.json, run_config.txt, .complete,
#                     task*/train.log, task*/**/log.txt (dist eval scores), eval/*/answers.jsonl,
#                     cl_results.json and predictions/ (CL-LoRA), pl/bal/pick logs
# What stays out: merged/ models, weight files (.safetensors .bin .pt .pth .ckpt .npy .npz .pkl)
# and anything over MAX_MB. Oversize files are listed in SKIPPED.txt instead of copied.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")"

LABEL=${1:-$(hostname -s)_$(date +%Y%m%d_%H%M)}
MAX_KB=$(( ${MAX_MB:-50} * 1024 ))
OUT=collected_logs/${LABEL}
[ ! -e "${OUT}" ] || { echo "${OUT} already exists, pick another label"; exit 1; }
mkdir -p "${OUT}"

copy () {  # $1=source file $2=destination path inside OUT
    mkdir -p "${OUT}/$(dirname "$2")"
    cp -p "$1" "${OUT}/$2"
}

n_root=0
for f in logs_*.log; do
    copy "$f" "$f"
    n_root=$((n_root + 1))
done

n_tok=0
for f in /tmp/tok_*.log; do
    copy "$f" "tmp/$(basename "$f")"
    n_tok=$((n_tok + 1))
done

n_res=0; n_skip=0
if [ -d results ]; then
    # merged/ is pruned whole; everything else is filtered file by file.
    KEEP=(-type f ! -name '*.safetensors' ! -name '*.bin' ! -name '*.pt' ! -name '*.pth'
          ! -name '*.ckpt' ! -name '*.npy' ! -name '*.npz' ! -name '*.pkl')
    while IFS= read -r -d '' f; do
        copy "$f" "$f"
        n_res=$((n_res + 1))
    done < <(find results -path '*/merged' -prune -o "${KEEP[@]}" -size -$((MAX_KB + 1))k -print0)
    while IFS= read -r -d '' f; do
        du -h "$f" >> "${OUT}/SKIPPED.txt"
        n_skip=$((n_skip + 1))
    done < <(find results -path '*/merged' -prune -o "${KEEP[@]}" -size +${MAX_KB}k -print0)
else
    echo "no results/ here, only the root logs are collected"
fi

echo "root logs:    ${n_root}"
echo "tokenize:     ${n_tok}"
echo "results:      ${n_res} files"
[ "${n_skip}" -eq 0 ] || echo "skipped:      ${n_skip} files over ${MAX_MB:-50} MB, see ${OUT}/SKIPPED.txt"
echo "folder:       ${OUT} ($(du -sh "${OUT}" | cut -f1))"
