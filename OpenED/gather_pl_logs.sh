#!/bin/bash
# Collect what pseudo-label precision/recall needs, for every run that pseudo-labeled, into pl_logs/
# (rebuilt on each call: it only holds copies) and pack it as pl_logs.tar.gz.
#
#   bash gather_pl_logs.sh               # ACE runs (the only corpus with an unstripped --oracle split)
#   DS=maven bash gather_pl_logs.sh      # another corpus' runs: files only, no P/R
#   DS=all bash gather_pl_logs.sh
#
# Per run <run> whose name holds _<ds>_ (repo layout kept):
#   data/stage_<run>/<t>_pl/train.jsonl, pl_stats.json   the task data with the pseudo-labels merged in
#   results/qwen3/ced/<run>/run_manifest.json            perm and data root
#   results/qwen3/ced/<run>/pl_task<t>.log               PL log, PL_QUALITY lines in the newer runs
# plus pl_logs/pl_quality_ace.log: tools/ced_pl_quality.py over the ACE runs whose oracle split
# data/ace_oracle_b10_perm<p> is here, P/R per task and pooled per config. Recomputing elsewhere
# also needs data/ace_b10_perm<p> and data/ace_oracle_b10_perm<p> (both in HF datht/processed-cl-ace).
# The _boost / _sd stage folders are not copied: they repeat the _pl data.
set -uo pipefail
cd "$(dirname "$0")"

DS=${DS:-ace}
OUT=pl_logs
R=results/qwen3/ced
PY=${PY:-$(command -v python || command -v python3)}
rm -rf "${OUT}" "${OUT}.tar.gz"
mkdir -p "${OUT}"

n=0; runs=(); ace_runs=()
for stage in data/stage_*/; do
    run=$(basename "${stage}"); run=${run#stage_}
    [ "${DS}" = "all" ] || [[ "${run}" == *_${DS}_* ]] || continue
    compgen -G "${stage}*_pl/train.jsonl" > /dev/null || continue
    for f in "${stage}"*_pl/train.jsonl "${stage}"*_pl/pl_stats.json \
             "${R}/${run}/run_manifest.json" "${R}/${run}"/pl_task*.log; do
        [ -f "${f}" ] || continue
        mkdir -p "${OUT}/$(dirname "${f}")"
        cp -p "${f}" "${OUT}/${f}"
        n=$((n + 1))
    done
    runs+=("${run}")
    p=$(sed -n 's/.*_perm\([0-9]\+\)_.*/\1/p' <<< "${run}")
    if [[ "${run}" == *_ace_* ]] && [ -f "${R}/${run}/run_manifest.json" ] \
            && [ -s "data/ace_oracle_b10_perm${p}/streams.json" ]; then
        ace_runs+=("${R}/${run}")
    fi
done
echo "runs:   ${#runs[@]} with pseudo-labels (DS=${DS}), ${n} files"
printf '  %s\n' "${runs[@]}" > "${OUT}/runs.txt"

if [ ${#ace_runs[@]} -gt 0 ]; then
    "${PY}" tools/ced_pl_quality.py "${ace_runs[@]}" > "${OUT}/pl_quality_ace.log" 2>&1 \
        || echo "ced_pl_quality.py failed, see ${OUT}/pl_quality_ace.log"
    echo "P/R:    ${OUT}/pl_quality_ace.log (${#ace_runs[@]} ACE runs)"
elif [[ "${DS}" == ace || "${DS}" == all ]]; then
    echo "P/R:    not computed: no ACE run with run_manifest.json and data/ace_oracle_b10_perm<p>"
fi

if command -v tar > /dev/null; then
    tar czf "${OUT}.tar.gz" "${OUT}"
else
    "${PY}" -c "import sys, tarfile; tarfile.open(sys.argv[1], 'w:gz').add(sys.argv[2])" "${OUT}.tar.gz" "${OUT}"
fi
echo "folder: ${OUT} ($(du -sh "${OUT}" | cut -f1)), packed: ${OUT}.tar.gz ($(du -sh "${OUT}.tar.gz" | cut -f1))"
