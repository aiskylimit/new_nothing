#! /bin/bash
# Starts only the GPU 7 queue of run.sh (same jobs, same order, same GPU and port) and returns; the runs
# keep going after logout. For rerunning that queue alone, e.g. after it failed; don't run it while
# run.sh's GPU 7 queue is still alive. Unpacks processed_data.zip itself when processed_data/ is missing or old.
# Console output: logs/<script>[_<dataset>].log. Training log with the RESULT lines:
# results/qwen3/general_domain/sft_*/<settings>/log.txt (teachers: sft_4B_<data>_<dataset>)
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

# same unpacking as run.sh: an old processed_data/ (EE without "unknown_args") is unpacked again from the zip
fresh () { grep -q '"unknown_args"' processed_data/general_domain/ee_v1/qwen/train.jsonl 2>/dev/null; }
if [ ! -f processed_data/general_domain/ner_v2/qwen/train_0.bin ] || ! fresh; then
    if [ ! -f processed_data.zip ]; then
        echo "missing processed_data.zip: fetch it with download.txt first"
        exit 1
    fi
    echo "unpacking processed_data.zip"
    python -c "import zipfile; zipfile.ZipFile('processed_data.zip').extractall('.')" || exit 1
fi
if ! fresh; then
    echo "processed_data.zip is the old one (EE without arguments): delete it and fetch it again with download.txt"
    exit 1
fi

for d in ee_v1 ee_v2 re_tacred re_fewrel; do
    if [ ! -f processed_data/general_domain/${d}/qwen/train_0.bin ]; then
        echo "missing processed_data/general_domain/${d}/qwen after unpacking processed_data.zip"
        exit 1
    fi
done

# SMOKE=1 bash run2.sh: each job trains one optimizer step (64 rows = one effective batch of 32, twice, the
# loop stops after the first step), saves, then scores 16 dev and 16 test rows. A torchrun wrapper appends
# the flags (argparse keeps the last value) and moves --save to results_smoke/; logs/smoke_*.log, port +1,
# so it can run next to the real queue.
PREFIX="" PORT_OFFSET=0
if [ "${SMOKE:-0}" = 1 ]; then
    PREFIX="smoke_" PORT_OFFSET=1
    mkdir -p logs/smoke_bin
    cat > logs/smoke_bin/torchrun <<EOF
#!/bin/bash
args=()
for a in "\$@"; do args+=("\${a/results\/qwen3/results_smoke/qwen3}"); done
exec $(command -v torchrun) "\${args[@]}" --train-num 64 --dev-num 16 --epochs 1
EOF
    chmod +x logs/smoke_bin/torchrun
    export PATH="${PWD}/logs/smoke_bin:${PATH}"
fi

# same as run.sh: the jobs run one after the other on the GPU, on torchrun port 29500 + 10 x GPU
QUEUE_LOGS=() QUEUE_PID=""
start () {  # $1 = GPU, then jobs: script path under scripts/general_domain/, optionally :<dataset>
    local gpu=$1; shift
    local mark=logs/.${PREFIX}gpu${gpu}_started
    local cmd=": > ${mark}; "
    for j in "$@"; do
        local s=${j%%:*} ds=""
        [ "${j}" != "${s}" ] && ds=${j#*:}
        local log=logs/${PREFIX}$(basename ${s})${ds:+_${ds}}.log
        cmd+="GPU=${gpu} PORT=$((29500 + 10 * gpu + PORT_OFFSET)) bash scripts/general_domain/${s}.sh ${ds} > ${log} 2>&1; "
        QUEUE_LOGS+=("${log}")
    done
    rm -f ${mark} "${QUEUE_LOGS[@]}"  # the queue rewrites them; removed so the console doesn't replay old runs
    nohup bash -c "${cmd}" > /dev/null 2>&1 &
    local pid=$!
    QUEUE_PID=${pid}
    # the runner hangs up the session as soon as this script exits; a queue still starting up (before nohup
    # has made it ignore SIGHUP) dies then without writing a single log, which is what kept hitting the GPU 7
    # queue, the last one started. Wait until the queue's own bash runs (from then on it ignores SIGHUP).
    for i in $(seq 50); do [ -f ${mark} ] && break; sleep 0.2; done
    if [ -f ${mark} ]; then
        echo "started gpu${gpu} (pid ${pid}): $*"
    else
        echo "gpu${gpu} queue did not start within 10 s"
    fi
}

start 7 ee/sft_teacher_4B_ee_v2:rams re/sft_teacher_4B_re_tacred \
    ee/sft_teacher_4B_ee_v1:ace ee/sft_teacher_4B_ee_v2:ace ee/sft_teacher_4B_ee_v1:geneva ee/sft_teacher_4B_ee_v2:geneva \
    ee/sft_student_0.6B_ee_v1 ee/sft_student_0.6B_ee_v2 re/sft_student_0.6B_re_fewrel

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"

# the logs also go to this console, until the queue ends. The queue runs on its own: if the runner stops this
# script (time limit, hang-up), only the console copy stops, the training and logs/ go on.
tail -n +1 -F --pid=${QUEUE_PID} "${QUEUE_LOGS[@]}" 2>/dev/null
echo "gpu7 queue finished"
