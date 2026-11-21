#! /bin/bash
# Two EventKD runs on ACE alone (one teacher, sfkl, constant KD 0.9, span loss 1.0, as eventkd_ace of run6.sh),
# each changing one thing of the span loss, ~0.5 h each on GPUs 0 and 1. Start once run6.sh is done.
#   eventkd_ace_l1   the span similarities compared by absolute difference (SPAN_DIST=l1, the EventKD paper's
#                    formula) instead of squared difference: the span loss was ~2-3% of the loss in run6
#   eventkd_ace_cka  span layers picked by CKA (student 16/24/27 <-> teacher 19/32/35, both before the final
#                    norm) instead of 25/28 <-> 33/36, whose 28 <-> 36 pair compares the student before its
#                    final norm with the teacher after it
# References (run6.sh, test at the best dev epoch, trigger / argument F1): eventkd_ace 74.14 / 47.08,
# sfkl_ace 74.62 / 48.04. One seed each, so differences under ~2 points are noise.
# The training logs now also print "train | span (...)": the weighted span loss, next to ds_loss.
# Console output: logs/<run>.log; training logs: results/qwen3/general_domain/bam_0.6B_4B_mt_v2_ace/<settings>/log.txt
# SMOKE=1 bash run8.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
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

echo "teacher (best dev checkpoint):"
lora=$(python tools/best_lora.py results/qwen3/general_domain/sft_4B_ee_v2_ace) || exit 1
echo "  ee_v2_ace: ${lora}"

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
ONE="DATASET=ace KD_SCHEDULE=constant KD_RATIO=0.9 KD_TYPE=sfkl W_SPAN=1.0"   # eventkd_ace of run6.sh

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run8_gpu${gpu}_started
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

start 0 eventkd_ace_l1 "${ONE} SPAN_DIST=l1 ${BAM}"
start 1 eventkd_ace_cka "${ONE} S_LAYERS='16 24 27' T_LAYERS='19 32 35' ${BAM}"

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/general_domain/bam_0.6B_4B_mt_v2_ace --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run8 queues finished"
