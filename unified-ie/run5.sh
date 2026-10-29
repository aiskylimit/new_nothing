#! /bin/bash
# EventKD on the multi-task student (mt_v2, all train rows): why BAM + EventKD (sfkl + span loss) did not
# beat plain BAM (fkl) in run4.sh, and settings that may help. 8 runs, 2 per GPU 4-7, each about 19-21 h, so
# about 40-42 h in all. The run4.sh runs on all rows are the reference (mean of EE, NER, RE):
#   BAM        fkl, anneal, no span          dev 62.64 test 63.45
#   EventKD    sfkl, anneal, span 1.0        dev 63.25 test 62.91
# GPU 4  ablation: sfkl without the span loss; fkl with the span loss (which of the two costs)
# GPU 5  span loss weight 0.1, 0.5 (instead of 1.0)
# GPU 6  fixed KD weight 0.5 instead of teacher annealing: BAM, EventKD
# GPU 7  EventKD with the span loss kept at full weight while KD anneals (SPAN_SCHEDULE=constant); EventKD
#        with the original span search (SPAN_LOCATE=legacy)
# Same teachers as run4.sh (best dev checkpoint of each of the 8). Same setup as run.sh (venv, PET_* unset,
# port 29500 + 10 x GPU). Console output: logs/<run>.log; training logs with the RESULT lines:
# results/qwen3/general_domain/bam_0.6B_4B_mt_v2/<run settings>/.../log.txt
# SMOKE=1 bash run5.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
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
EVENTKD="KD_TYPE=sfkl W_SPAN=1.0"

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run5_gpu${gpu}_started
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

start 4 bam_sfkl "KD_TYPE=sfkl ${BAM}" \
        bam_fkl_span "W_SPAN=1.0 ${BAM}"
start 5 bam_eventkd_span0.1 "KD_TYPE=sfkl W_SPAN=0.1 ${BAM}" \
        bam_eventkd_span0.5 "KD_TYPE=sfkl W_SPAN=0.5 ${BAM}"
start 6 bam_kd0.5 "KD_SCHEDULE=constant KD_RATIO=0.5 ${BAM}" \
        bam_eventkd_kd0.5 "KD_SCHEDULE=constant KD_RATIO=0.5 ${EVENTKD} ${BAM}"
start 7 bam_eventkd_spanconst "SPAN_SCHEDULE=constant ${EVENTKD} ${BAM}" \
        bam_eventkd_legacy "SPAN_LOCATE=legacy ${EVENTKD} ${BAM}"

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/general_domain/bam_0.6B_4B_mt_v2 --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run5 queues finished"
