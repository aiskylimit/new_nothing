#! /bin/bash
# The general_domain students trained on one mt_v2 dataset (so one task) at a time, with every baseline of the
# bio/Vietnamese tables (run10.sh, run11.sh), so the three domains share one setting. One teacher per dataset
# (run7.sh), constant KD weight 0.9, no span loss except EventKD:
#   sft      CE only (scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh with DATASET=<ds>)
#   fkl      forward KL, standard KD (the "KD" row)
#   amid     AMiD (alpha 0.5, lambda 0.5, forward KL from the teacher to the mixture)
#   eventkd  sfkl + span loss 1.0 (EventKD); sfkl = the same without span loss (SFKL)
#   rkl      reverse KL; srkl = skewed reverse KL, skew 0.1 (DistiLLM's loss); csd = CSD
# eventkd and sfkl of ace, conll03, fewrel are in run6.sh (same setting), so not run again. 58 runs.
# Datasets: EE ace, geneva, maven, rams; NER conll03, ontonotes; RE tacred, fewrel.
# Order: sft, fkl, amid, eventkd first (~20 h), then sfkl, rkl, srkl, csd. The jobs are split over GPUs 0-1
# by run time, estimated from run6 (sfkl: ace 0.3 h, geneva 0.25, rams 0.5, tacred 0.35, conll03 1, maven 1.9,
# fewrel 2.3, ontonotes 4.2; SFT ~0.7x): ~38 h per GPU, +-30%.
# Console output: logs/general_<method>_<ds>.log; training logs: results/qwen3/general_domain/
# {sft_0.6B_mt_v2_<ds>, bam_0.6B_4B_mt_v2_<ds>/<settings>}/log.txt
# SMOKE=1 bash run12.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
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

# data: label maps from run.sh's unpack of processed_data.zip; mt_v2 from mt_v2.zip (download.txt)
for f in data/general_domain/ee_label_map.tsv data/general_domain/ner_label_map.tsv; do
    if [ ! -f ${f} ]; then
        echo "missing ${f}: run run.sh once to unpack processed_data.zip"
        exit 1
    fi
done
if [ ! -f processed_data/general_domain/mt_v2/qwen/train_0.bin ]; then
    python tools/unpack_data.py mt_v2 || exit 1
fi

echo "teachers (best dev checkpoint, from run7.sh):"
for t in sft_4B_ee_v2_ace sft_4B_ee_v2_geneva sft_4B_ee_v2_maven sft_4B_ee_v2_rams sft_4B_ner_v2_conll03          sft_4B_ner_v2_ontonotes sft_4B_re_tacred sft_4B_re_fewrel; do
    lora=$(python tools/best_lora.py results/qwen3/general_domain/${t}) || exit 1
    echo "  ${t}: ${lora}"
done

PREFIX="" PORT_OFFSET=0
if [ "${SMOKE:-0}" = 1 ]; then
    # a torchrun wrapper appends the smoke flags (argparse keeps the last value) and moves only the --save
    # dir to results_smoke/ (the teacher paths also contain results/qwen3 and must stay)
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

SFT="bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh"
BAM="bash scripts/general_domain/bam_0.6B_4B.sh mt_v2"
ONE="KD_SCHEDULE=constant KD_RATIO=0.9"     # single-teacher setting of run6.sh and run10.sh

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run12_gpu${gpu}_started
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

jobs () {  # $1 = GPU, then <method>:<dataset> ..., run in that order
    local gpu=$1; shift
    local args=() m ds
    for x in "$@"; do
        m=${x%%:*} ds=${x#*:}
        case ${m} in
            sft)     args+=(general_sft_${ds} "DATASET=${ds} ${SFT}");;
            eventkd) args+=(general_eventkd_${ds} "DATASET=${ds} ${ONE} KD_TYPE=sfkl W_SPAN=1.0 ${BAM}");;
            *)       args+=(general_${m}_${ds} "DATASET=${ds} ${ONE} KD_TYPE=${m} ${BAM}");;
        esac
    done
    start ${gpu} "${args[@]}"
}

# split by estimated run time (see the header); sft, fkl, amid, eventkd before the rest on both GPUs
jobs 0 sft:ace sft:tacred sft:conll03 sft:fewrel fkl:ace fkl:geneva fkl:rams fkl:tacred fkl:conll03 \
    fkl:fewrel amid:ace amid:geneva amid:rams amid:tacred amid:conll03 amid:maven amid:ontonotes \
    eventkd:ontonotes rkl:ace rkl:geneva rkl:rams rkl:tacred rkl:conll03 rkl:maven rkl:ontonotes \
    srkl:fewrel csd:ace csd:geneva csd:rams csd:tacred csd:conll03 csd:maven csd:ontonotes
jobs 1 sft:geneva sft:rams sft:maven sft:ontonotes fkl:maven fkl:ontonotes amid:fewrel eventkd:geneva \
    eventkd:rams eventkd:tacred eventkd:maven sfkl:geneva sfkl:rams sfkl:tacred sfkl:maven \
    sfkl:ontonotes rkl:fewrel srkl:ace srkl:geneva srkl:rams srkl:tacred srkl:conll03 srkl:maven \
    srkl:ontonotes csd:fewrel

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/general_domain/{sft_0.6B_mt_v2_,bam_0.6B_4B_mt_v2_}* --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run12 queues finished"
