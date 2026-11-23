#! /bin/bash
# KD baselines on the biomedical and Vietnamese datasets, one dataset (so one task) at a time, same setting as
# sfkl_<ds> of run10.sh (one run9 teacher, constant KD weight 0.9, no span loss), only the KD loss changes:
#   fkl_<ds>    forward KL, standard KD (the "KD" row)
#   rkl_<ds>    reverse KL (RKL)
#   srkl_<ds>   skewed reverse KL, skew 0.1: the DistiLLM loss, without DistiLLM's student-generated outputs
#   csd_<ds>    CSD
#   amid_<ds>   AMiD (alpha-mixture assistant distribution, the aailab-kaist/AMiD defaults: alpha 0.5,
#               lambda 0.5, forward KL from the teacher to the mixture), ported from OpenED
# FDD is not in the code yet. SFT, sfkl (SFKL) and EventKD of the same datasets are in run10.sh.
#   specific_domain  NER bc5cdr, RE ddi, EE phee
#   vietnamese       NER vlsp2021, RE vlsp2020, EE bkee
# Times from run10 (sfkl, per run): bc5cdr 45 min, ddi 30, phee 20, vlsp2021 70, vlsp2020 30, bkee 60.
# GPU 0: the biomedical datasets and vlsp2020 (~10.5 h), GPU 1: vlsp2021 and bkee (~10.5 h), one loss after the
# other, so every loss has all its datasets before the next one starts.
# Console output: logs/<prefix>_<loss>_<ds>.log; training logs: results/qwen3/<domain>/bam_0.6B_4B_all_<ds>/<settings>/log.txt
# SMOKE=1 bash run11.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
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

# data: unpacked by run9.sh
for f in data/{specific_domain,vietnamese}/label_map.tsv processed_data/{specific_domain,vietnamese}/all/qwen/train_0.bin; do
    [ -f ${f} ] || { echo "missing ${f}: run run9.sh first"; exit 1; }
done

echo "teachers (best dev checkpoint, from run9.sh):"
for t in specific_domain/sft_4B_ner_bc5cdr specific_domain/sft_4B_re_ddi specific_domain/sft_4B_ee_phee \
         vietnamese/sft_4B_ner_vlsp2021 vietnamese/sft_4B_re_vlsp2020 vietnamese/sft_4B_ee_bkee; do
    lora=$(python tools/best_lora.py results/qwen3/${t}) || exit 1
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

BAM="bash scripts/general_domain/bam_0.6B_4B.sh all"
ONE="KD_SCHEDULE=constant KD_RATIO=0.9"     # single-teacher setting of run10.sh, no span loss

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run11_gpu${gpu}_started
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

jobs () {  # $1 = GPU, then <prefix>:<domain>:<dataset> ...; every loss over all of them, then the next loss
    local gpu=$1; shift
    local args=()
    for loss in fkl rkl srkl csd amid; do
        for x in "$@"; do
            IFS=: read p d ds <<< "${x}"
            args+=(${p}_${loss}_${ds} "DOMAIN=${d} DATASET=${ds} ${ONE} KD_TYPE=${loss} ${BAM}")
        done
    done
    start ${gpu} "${args[@]}"
}

B=specific_domain V=vietnamese
jobs 0 bio:${B}:bc5cdr bio:${B}:ddi bio:${B}:phee vn:${V}:vlsp2020
jobs 1 vn:${V}:vlsp2021 vn:${V}:bkee

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/specific_domain results/qwen3/vietnamese --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run11 queues finished"
