#! /bin/bash
# Starts only the GPU 7 queue of run.sh (same jobs, same order, same GPU and port) and returns; the runs
# keep going after logout. For rerunning that queue alone, e.g. after it failed; don't run it while
# run.sh's GPU 7 queue is still alive. Needs processed_data unpacked (run.sh does that).
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

for d in ee_v1 ee_v2 re_tacred re_fewrel; do
    if [ ! -f processed_data/general_domain/${d}/qwen/train_0.bin ]; then
        echo "missing processed_data/general_domain/${d}/qwen: run run.sh once to unpack processed_data.zip"
        exit 1
    fi
done
if ! grep -q '"unknown_args"' processed_data/general_domain/ee_v1/qwen/train.jsonl; then
    echo "processed_data/general_domain/ee_v1 is the old EE data (no arguments): run run.sh once to unpack the new zip"
    exit 1
fi

# same as run.sh: the jobs run one after the other on the GPU, on torchrun port 29500 + 10 x GPU
start () {  # $1 = GPU, then jobs: script path under scripts/general_domain/, optionally :<dataset>
    local gpu=$1; shift
    local cmd=""
    for j in "$@"; do
        local s=${j%%:*} ds=""
        [ "${j}" != "${s}" ] && ds=${j#*:}
        cmd+="GPU=${gpu} PORT=$((29500 + 10 * gpu)) bash scripts/general_domain/${s}.sh ${ds} > logs/$(basename ${s})${ds:+_${ds}}.log 2>&1; "
    done
    nohup bash -c "${cmd}" > /dev/null 2>&1 &
    echo "started gpu${gpu} (pid $!): $*"
}

start 7 ee/sft_teacher_4B_ee_v2:rams re/sft_teacher_4B_re_tacred \
    ee/sft_teacher_4B_ee_v1:ace ee/sft_teacher_4B_ee_v2:ace ee/sft_teacher_4B_ee_v1:geneva ee/sft_teacher_4B_ee_v2:geneva \
    ee/sft_student_0.6B_ee_v1 ee/sft_student_0.6B_ee_v2 re/sft_student_0.6B_re_fewrel

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"
