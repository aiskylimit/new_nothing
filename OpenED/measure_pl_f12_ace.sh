#!/usr/bin/env bash
# Pseudo-label precision / recall of the ACE "Ours" runs (f12_pl, A40_3, 5 perms), which kept their
# pseudo-labelled data on A40_3 only. The LoRA adapters of every task are on HF
# (datht/ced-eventkd-artifacts, A40_3/f12_pl[_perm<p>]), so this rebuilds each run's teachers and
# reruns only the pseudo-labelling step:
#   teacher of task t = Qwen3-0.6B + the last adapter of tasks 0..t-1, merged one after another
#   (tools/merge_lora.py, bf16, as run_ced_v2.sh merges), then tools/ced_pseudo_label.py on
#   data/ace_b10_perm<p>/<t> with the PL rules f12_pl ran with (July, before H1/H2 existed):
#   old-type events, trigger verbatim in the sentence, lexicon filter, exact duplicates dropped,
#   no conflict dedup (--conflict-dedup 0), no confidence filter. Greedy decoding, so the labels
#   match the original run up to bf16 merge rounding.
# P / R come from the PL_QUALITY lines (precision and recall against the stripped records of the
# --oracle split data/ace_oracle_b10_perm<p>, built here when missing; HF_TOKEN from .env).
# Nothing is trained. Work files in pl_f12_ace/ (one merged teacher kept at a time).
#
#   bash measure_pl_f12_ace.sh                 # GPU 7
#   GPU=5 bash measure_pl_f12_ace.sh
#   PERMS="0" bash measure_pl_f12_ace.sh
# Result: logs/pl_f12_ace_summary.txt (per task, per perm, pooled), logs/pl_f12_ace.log (everything).
set -uo pipefail
cd "$(dirname "$0")"
GPU=${GPU:-7}
PERMS=${PERMS:-"0 1 2 3 4"}
W=pl_f12_ace
LOG=logs/pl_f12_ace.log
mkdir -p logs "${W}"
log () { echo "[pl_f12 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then set +u; source "${VENV}/bin/activate"; set -u; fi
PY=${PY:-$(command -v python || command -v python3)}
[ -f .env ] && [ -z "${HF_TOKEN:-}" ] && { set -a; . ./.env; set +a; }
BASE=models/Qwen3-0.6B
[ -f "${BASE}/config.json" ] || { log "no ${BASE}"; exit 1; }

# ---------------------------------------------------------------- data: task splits and oracle splits
missing=""
for p in ${PERMS}; do
    [ -s "data/ace_b10_perm${p}/streams.json" ] || { log "no data/ace_b10_perm${p}"; exit 1; }
    [ -s "data/ace_oracle_b10_perm${p}/streams.json" ] || missing+=" ${p}"
done
if [ -n "${missing}" ]; then
    log "building data/ace_oracle_b10_perm{${missing# }}"
    # shellcheck disable=SC2086
    OPENED_BASE=$(pwd) HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "${PY}" tools/build_ced_perms.py \
        --cap 10 --oracle --perms ${missing} --out-prefix ace_oracle_b10_perm >> "${LOG}" 2>&1 \
        || { log "oracle build failed, see ${LOG}"; exit 1; }
fi

# ---------------------------------------------------------------- adapters: the last checkpoint of tasks 0-3
log "downloading the f12_pl adapters of tasks 0-3"
HF_HUB_OFFLINE=0 "${PY}" - "${W}/adapters" ${PERMS} >> "${LOG}" 2>&1 <<'EOF' || { log "adapter download failed, see ${LOG}"; exit 1; }
import os, re, sys
from huggingface_hub import HfApi, hf_hub_download
out, perms = sys.argv[1], sys.argv[2:]
repo = "datht/ced-eventkd-artifacts"
files = HfApi().list_repo_files(repo, repo_type="dataset")
for p in perms:
    run = "f12_pl" + (f"_perm{p}" if p != "0" else "")
    for t in range(4):
        steps = {}
        for f in files:
            m = re.match(rf"A40_3/{run}/task{t}/(?:.+/)?(\d+)/adapter_model\.bin$", f)
            if m:
                steps[int(m.group(1))] = f.rsplit("/", 1)[0]
        assert steps, f"no adapter for {run} task{t}"
        src = steps[max(steps)]
        dst = os.path.join(out, f"perm{p}", f"task{t}")
        for name in ("adapter_model.bin", "adapter_config.json"):
            if not os.path.exists(os.path.join(dst, name)):
                os.makedirs(dst, exist_ok=True)
                path = hf_hub_download(repo, f"{src}/{name}", repo_type="dataset", local_dir=os.path.join(out, "hf"))
                os.replace(path, os.path.join(dst, name))
        print(f"{run} task{t}: {src}")
EOF

# ---------------------------------------------------------------- per perm: merge chain, pseudo-label tasks 1-4
for p in ${PERMS}; do
    teacher=${BASE}
    for t in 1 2 3 4; do
        merged=${W}/teacher_perm${p}_task${t}
        if [ ! -f "${merged}/config.json" ]; then
            log "perm${p}: merging the task$((t - 1)) adapter -> ${merged}"
            CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" tools/merge_lora.py --base-model-path "${teacher}" \
                --peft-path "${W}/adapters/perm${p}/task$((t - 1))" --out "${merged}" >> "${LOG}" 2>&1 \
                || { log "merge failed (perm${p} task${t}), see ${LOG}"; exit 1; }
        fi
        out=${W}/pl/perm${p}/${t}_pl
        if [ ! -f "${out}/pl_stats.json" ]; then
            log "perm${p} task${t}: pseudo-labelling"
            CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PY}" tools/ced_pseudo_label.py \
                --teacher "${merged}" --data-dir "data/ace_b10_perm${p}/${t}" \
                --streams "data/ace_b10_perm${p}/streams.json" --task-id "${t}" --out "${out}" \
                --batch-size 64 --conflict-dedup 0 --lexicon-filter 1 --conf-filter none --gen-backend hf \
                --oracle-dir "data/ace_oracle_b10_perm${p}/${t}" > "${W}/pl_perm${p}_task${t}.log" 2>&1 \
                || { log "pseudo-labelling failed (perm${p} task${t}), see ${W}/pl_perm${p}_task${t}.log"; exit 1; }
            grep PL_QUALITY "${W}/pl_perm${p}_task${t}.log" | tee -a "${LOG}"
        fi
        [ "${teacher}" != "${BASE}" ] && rm -rf "${teacher}"   # keep one merged teacher at a time
        teacher=${merged}
    done
    rm -rf "${teacher}"
done

# ---------------------------------------------------------------- summary
"${PY}" - "${W}" ${PERMS} > logs/pl_f12_ace_summary.txt 2>&1 <<'EOF'
import json, os, statistics as st, sys
w, perms = sys.argv[1], sys.argv[2:]
pool = {"added": 0, "hit": 0, "stripped": 0}
per_perm = []
print("perm task  added    hit stripped      P      R")
for p in perms:
    c = {"added": 0, "hit": 0, "stripped": 0}
    for t in range(1, 5):
        q = json.load(open(os.path.join(w, "pl", f"perm{p}", f"{t}_pl", "pl_stats.json"))).get("pl_quality")
        if q is None:
            print(f"{p:>4} {t:>4}  no PL_QUALITY (oracle rows did not line up?)")
            continue
        k = q["kept"]
        print(f"{p:>4} {t:>4} {k['added']:6d} {k['hit']:6d} {q['stripped']:8d} {k['precision']:6.2f} {k['recall']:6.2f}")
        for x, v in (("added", k["added"]), ("hit", k["hit"]), ("stripped", q["stripped"])):
            c[x] += v
            pool[x] += v
    P, R = 100 * c["hit"] / max(c["added"], 1), 100 * c["hit"] / max(c["stripped"], 1)
    per_perm.append((P, R))
    print(f"perm{p} pooled: P {P:.2f} R {R:.2f}")
ms = lambda xs: f"{st.mean(xs):.2f}" + (f" +- {st.stdev(xs):.2f}" if len(xs) > 1 else "")
print(f"\nmean over perms: P {ms([x[0] for x in per_perm])}  R {ms([x[1] for x in per_perm])}")
print(f"pooled over perms and tasks (as tools/ced_pl_quality.py): "
      f"P {100 * pool['hit'] / max(pool['added'], 1):.2f}  R {100 * pool['hit'] / max(pool['stripped'], 1):.2f}")
EOF
cat logs/pl_f12_ace_summary.txt | tee -a "${LOG}"
log "done: logs/pl_f12_ace_summary.txt"
