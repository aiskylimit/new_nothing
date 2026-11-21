#! /bin/bash
# The biomedical (specific_domain) and Vietnamese (vietnamese) experiments, separate from general_domain:
# one Qwen3-4B LoRA teacher per source dataset, then the multi-task Qwen3-0.6B students on <domain>/all:
#   SFT      CE only (scripts/domains/sft_student_0.6B.sh), the baseline
#   EventKD  BAM multi-teacher KD + sfkl + span loss 1.0, KD weight annealed 1 -> 0, as bam_eventkd_mt_v2 on
#            general_domain (KD_TYPE=sfkl W_SPAN=1.0 bash scripts/general_domain/bam_0.6B_4B.sh, DOMAIN=<domain>)
#   specific_domain  NER bc5cdr, RE ddi, EE phee (13,270 train rows)
#   vietnamese       NER vlsp2021 (1/3 of its train), RE vlsp2020, EE bkee (16,127 train rows)
# One domain per GPU: its 3 teachers, then SFT, then EventKD (needs the teachers).
# Rough times from the general_domain runs (rows x epochs per hour): GPU 0 (biomedical) ~4.5 h,
# GPU 1 (Vietnamese) ~5.5 h, +-30%.
# Needs, after a fresh clone: the venv (install.sh), download.txt (models, specific_domain.zip, vietnamese.zip).
# Teachers stay on the server (no HF upload from here).
# Console output: logs/<run>.log; training logs: results/qwen3/<domain>/<run>/<settings>/log.txt
# SMOKE=1 bash run9.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
# into results_smoke/ and logs/smoke_*.log on port +1.
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

# data: <domain>.zip (download.txt) holds data/<domain>/label_map.tsv and processed_data/<domain>/*/qwen.
# Python's zipfile: the image has no unzip binary.
for d in specific_domain vietnamese; do
    if [ ! -f data/${d}/label_map.tsv ] || [ ! -f processed_data/${d}/all/qwen/train_0.bin ]; then
        if [ ! -f ${d}.zip ]; then
            echo "missing ${d}.zip: fetch it with download.txt first"
            exit 1
        fi
        echo "unpacking ${d}.zip"
        python -c "import zipfile; zipfile.ZipFile('${d}.zip').extractall('.')" || exit 1
    fi
done
for f in processed_data/{specific_domain,vietnamese}/{ner,re,ee,all}/qwen/train_0.bin; do
    [ -f ${f} ] || { echo "missing ${f} after unpacking"; exit 1; }
done

PREFIX="" PORT_OFFSET=0
if [ "${SMOKE:-0}" = 1 ]; then
    # a torchrun wrapper appends the smoke flags (argparse keeps the last value) and moves the --save dir
    # to results_smoke/
    PREFIX="smoke_" PORT_OFFSET=1
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

T="bash scripts/domains/sft_teacher_4B.sh"
SFT="bash scripts/domains/sft_student_0.6B.sh"
EVENTKD="KD_TYPE=sfkl W_SPAN=1.0 bash scripts/general_domain/bam_0.6B_4B.sh all"
if [ "${SMOKE:-0}" = 1 ]; then
    # the smoke EventKD runs read the smoke teachers in results_smoke/
    sed 's#results/qwen3#results_smoke/qwen3#' scripts/general_domain/bam_0.6B_4B.sh > logs/smoke_bin/bam.sh
    EVENTKD="KD_TYPE=sfkl W_SPAN=1.0 bash logs/smoke_bin/bam.sh all"
fi

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run9_gpu${gpu}_started
    local cmd=": > ${mark}; " names=""
    while [ $# -gt 0 ]; do
        local log=logs/${PREFIX}$1.log
        cmd+="GPU=${gpu} PORT=$((29600 + 10 * gpu + PORT_OFFSET)) $2 > ${log} 2>&1; "
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

B=specific_domain V=vietnamese
start 0 bio_teacher_ner_bc5cdr "${T} ${B} ner bc5cdr" bio_teacher_re_ddi "${T} ${B} re ddi" \
        bio_teacher_ee_phee "${T} ${B} ee phee" bio_sft "${SFT} ${B}" bio_eventkd "DOMAIN=${B} ${EVENTKD}"
start 1 vn_teacher_ner_vlsp2021 "${T} ${V} ner vlsp2021" vn_teacher_re_vlsp2020 "${T} ${V} re vlsp2020" \
        vn_teacher_ee_bkee "${T} ${V} ee bkee" vn_sft "${SFT} ${V}" vn_eventkd "DOMAIN=${V} ${EVENTKD}"

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/specific_domain results/qwen3/vietnamese --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run9 queues finished"
