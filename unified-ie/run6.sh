#! /bin/bash
# Three things, about 27-30 h per GPU. Start after run7.sh has trained the 8 teachers again (server reset of
# 2026-11-04); the references below came from the old teachers.
# 1. EventKD without BAM, one dataset per task (EE ACE, NER CoNLL03, RE FewRel, where the student is ~7 points
#    under its teacher): the student learns only that dataset's rows from its one teacher
#    (DATASET=<name>), constant KD weight 0.9, sfkl, as in the EventKD paper (span layers 25/28 <-> 33/36 as
#    published). Each dataset twice: with the span loss (EventKD) and without it (sfkl only), so the gain of
#    the span loss per dataset can be read off. 6 runs on GPUs 4-5 (ACE ~0.5 h, CoNLL03 ~2 h, FewRel ~4.5 h each).
# 2. Span loss on the layers the CKA measure picked (Kaggle, base models): student 16/24/27 <-> teacher
#    19/32/35, no last layer (the 28 <-> 36 pair matched worst and made ~90% of the span loss). On the best
#    multi-task setting so far (fkl, constant KD 0.5, test mean 63.99). Those pairs start ~12x closer than the
#    old ones, so the span loss is ~12x smaller: weight 1.0 and 10.
# 3. Multi-task optimization (--mto, mto.py): the EE, NER and RE losses each a token mean over their own rows,
#    combined by plain mean (equal) or FAMO weights (famo), for the SFT student (does balancing the tasks
#    alone help?) and for BAM (anneal, fkl; does it add to multi-teacher KD?). SFT ~7-9 h, BAM ~20-22 h
#    (famo scores each step's micro-batches again after the update: one more student forward each).
#    References: SFT 62.35, BAM 63.45 (test mean).
# Multi-task references (fkl, no span, test mean of EE, NER, RE): BAM anneal 63.45, constant KD 0.5 63.99.
# Same teachers and setup as run5.sh. Console output: logs/<run>.log; training logs with the RESULT lines:
# results/qwen3/general_domain/bam_0.6B_4B_mt_v2[_<dataset>]/<run settings>/.../log.txt, sft_0.6B_mt_v2_mto<mode>/...
# SMOKE=1 bash run6.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
# into results_smoke/ and logs/smoke_*.log on port +1.
cd "$(dirname "$0")"
mkdir -p logs

# every run needs the unified-ie venv (peft etc.); the image's own python3.10 has no peft
VENV=/mnt/local/uvenvs/unified-ie
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: ${VENV} does not exist" >&2
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

# the 8 teachers: stop now rather than in 4 queues if one has no finished epoch
echo "teachers (best dev checkpoint):"
for r in ee_v2_ace ee_v2_geneva ee_v2_maven ee_v2_rams ner_v2_conll03 ner_v2_ontonotes re_tacred re_fewrel; do
    lora=$(python tools/best_lora.py results/qwen3/general_domain/sft_4B_${r}) || exit 1
    echo "  ${r}: ${lora}"
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

BAM="bash scripts/general_domain/bam_0.6B_4B.sh mt_v2"
ONE="KD_SCHEDULE=constant KD_RATIO=0.9 KD_TYPE=sfkl"     # single-teacher EventKD setting
NEW_LAYERS="S_LAYERS='16 24 27' T_LAYERS='19 32 35'"
KD05="KD_SCHEDULE=constant KD_RATIO=0.5"
SFT="bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh"

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run6_gpu${gpu}_started
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

start 4 eventkd_fewrel "DATASET=fewrel ${ONE} W_SPAN=1.0 ${BAM}" sfkl_fewrel "DATASET=fewrel ${ONE} ${BAM}" \
        bam_mto_equal "MTO=equal ${BAM}"
start 5 eventkd_ace "DATASET=ace ${ONE} W_SPAN=1.0 ${BAM}" sfkl_ace "DATASET=ace ${ONE} ${BAM}" \
        eventkd_conll03 "DATASET=conll03 ${ONE} W_SPAN=1.0 ${BAM}" sfkl_conll03 "DATASET=conll03 ${ONE} ${BAM}" \
        bam_mto_famo "MTO=famo ${BAM}"
start 6 sft_mto_equal "MTO=equal ${SFT}" \
        bam_kd0.5_span10_newlayers "${NEW_LAYERS} W_SPAN=10 ${KD05} ${BAM}"
start 7 sft_mto_famo "MTO=famo ${SFT}" \
        bam_kd0.5_span1_newlayers "${NEW_LAYERS} W_SPAN=1.0 ${KD05} ${BAM}"

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/general_domain/{bam_0.6B_4B_mt_v2*,sft_0.6B_mt_v2_mto*} --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run6 queues finished"
