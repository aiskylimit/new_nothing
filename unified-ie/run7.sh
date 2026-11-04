#! /bin/bash
# After the 2026-11-04 server reset (results/ gone): train the 8 Qwen3-4B LoRA teachers again, as the ones
# run4.sh/run5.sh used (tools/best_lora.py, best dev epoch of every run of that teacher). Those had:
#   5 epochs: ACE (best epoch 3, dev 75.45), GENEVA (5, 69.57), MAVEN (4, 65.61), RAMS (4, 74.52), TACRED (5, 39.39)
#   3 epochs: CoNLL03 (3, 97.64), OntoNotes (2, 89.29), FewRel (3, 49.19)
# Here all 8 get 5 epochs (the scripts' default; best_lora.py still picks the best dev epoch). Compare the
# new dev/test scores with those above before rerunning the KD references.
# Each queue then copies its teachers' best checkpoints to HF datht/unified-ie (tools/upload_teachers.py,
# HF_TOKEN or a cached login). Teachers only: about 4.5 h (OntoNotes the longest); run6.sh comes after.
# Needs, after a fresh clone: the venv (install.sh), download.txt (models, processed_data.zip, mt_v2.zip).
# Console output: logs/<run>.log; training logs: results/qwen3/general_domain/sft_4B_*/<settings>/log.txt.
# SMOKE=1 bash run7.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
# into results_smoke/ and logs/smoke_*.log on port +1; nothing is uploaded.
cd "$(dirname "$0")"
mkdir -p logs

# every run needs the unified-ie venv (peft etc.); the image's own python3.10 has no peft
VENV=/mnt/local/uvenvs/unified-ie
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: ${VENV} does not exist: run install.sh" >&2
    exit 1
fi
source "${VENV}/bin/activate"

# the pod's PET_* rendezvous variables would override --master_port (see run.sh)
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done

# data: tokenized folders + label maps from processed_data.zip, mt_v2 from mt_v2.zip (both via download.txt).
# Python's zipfile: the image has no unzip binary.
if [ ! -f data/general_domain/ner_label_map.tsv ] || [ ! -f processed_data/general_domain/ner_v2/qwen/train_0.bin ]; then
    if [ ! -f processed_data.zip ]; then
        echo "missing processed_data.zip: fetch it with download.txt first"
        exit 1
    fi
    echo "unpacking processed_data.zip"
    python -c "import zipfile; zipfile.ZipFile('processed_data.zip').extractall('.')" || exit 1
fi
for d in ee_v2 ner_v2 re_tacred re_fewrel; do
    if [ ! -f processed_data/general_domain/${d}/qwen/train_0.bin ]; then
        echo "missing processed_data/general_domain/${d}/qwen after unpacking processed_data.zip"
        exit 1
    fi
done
for f in data/general_domain/ee_label_map.tsv data/general_domain/ner_label_map.tsv; do
    [ -f ${f} ] || { echo "missing ${f} after unpacking processed_data.zip"; exit 1; }
done
if [ ! -f processed_data/general_domain/mt_v2/qwen/train_0.bin ]; then
    python tools/unpack_data.py mt_v2 || exit 1
fi

PREFIX="" PORT_OFFSET=0 UPLOAD="python tools/upload_teachers.py"
if [ "${SMOKE:-0}" = 1 ]; then
    # a torchrun wrapper appends the smoke flags (argparse keeps the last value) and moves the --save dir
    # to results_smoke/
    PREFIX="smoke_" PORT_OFFSET=1 UPLOAD="echo smoke, not uploading"
    mkdir -p logs/smoke_bin
    cat > logs/smoke_bin/torchrun <<EOF
#!/bin/bash
args=() prev=""
for a in "\$@"; do
    [ "\${prev}" = "--save" ] && a="\${a/results\/qwen3/results_smoke/qwen3}"
    args+=("\${a}") prev="\${a}"
done
exec $(command -v torchrun) "\${args[@]}" --train-num 64 --dev-num 16 --epochs 1
EOF
    chmod +x logs/smoke_bin/torchrun
    export PATH="${PWD}/logs/smoke_bin:${PATH}"
fi

EE="EPOCHS=5 bash scripts/general_domain/ee/sft_teacher_4B_ee_v2.sh"
NER="EPOCHS=5 bash scripts/general_domain/ner/sft_teacher_4B_ner.sh"
TACRED="EPOCHS=5 bash scripts/general_domain/re/sft_teacher_4B_re_tacred.sh"
FEWREL="EPOCHS=5 bash scripts/general_domain/re/sft_teacher_4B_re_fewrel.sh"
R=results/qwen3/general_domain

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run7_gpu${gpu}_started
    local cmd=": > ${mark}; " names=""
    while [ $# -gt 0 ]; do
        local log=logs/${PREFIX}$1.log
        cmd+="GPU=${gpu} PORT=$((29500 + 10 * gpu + PORT_OFFSET)) $2 > ${log} 2>&1; "
        QUEUE_LOGS+=("${log}") names+=" $1"
        rm -f ${log}
        shift 2
    done
    rm -f ${mark}
    nohup bash -c "${cmd}" > /dev/null 2>&1 &
    local pid=$!
    QUEUE_PIDS+=(${pid})
    # the runner hangs up the session as soon as this script exits; a queue still starting up dies then
    # without a log (see run.sh). Wait until the queue's own bash runs.
    for i in $(seq 50); do [ -f ${mark} ] && break; sleep 0.2; done
    if [ -f ${mark} ]; then
        echo "started gpu${gpu} (pid ${pid}):${names}"
    else
        echo "gpu${gpu} queue did not start within 10 s"
    fi
}

# 5 epochs, rows x epochs: OntoNotes ~4.5 h, FewRel ~3 h, MAVEN ~2.5 h, CoNLL03 ~1.2 h, the rest < 0.6 h
start 4 sft_teacher_4B_ner_ontonotes "${NER} ontonotes" \
        upload_teachers_gpu4 "${UPLOAD} ${R}/sft_4B_ner_v2_ontonotes"
start 5 sft_teacher_4B_re_fewrel "${FEWREL}" sft_teacher_4B_ee_v2_geneva "${EE} geneva" \
        upload_teachers_gpu5 "${UPLOAD} ${R}/sft_4B_re_fewrel ${R}/sft_4B_ee_v2_geneva"
start 6 sft_teacher_4B_ee_v2_maven "${EE} maven" sft_teacher_4B_ee_v2_rams "${EE} rams" \
        sft_teacher_4B_ee_v2_ace "${EE} ace" \
        upload_teachers_gpu6 "${UPLOAD} ${R}/sft_4B_ee_v2_maven ${R}/sft_4B_ee_v2_rams ${R}/sft_4B_ee_v2_ace"
start 7 sft_teacher_4B_ner_conll03 "${NER} conll03" sft_teacher_4B_re_tacred "${TACRED}" \
        upload_teachers_gpu7 "${UPLOAD} ${R}/sft_4B_ner_v2_conll03 ${R}/sft_4B_re_tacred"

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' ${R}/sft_4B_* --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run7 queues finished"
