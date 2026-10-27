#! /bin/bash
# Starts the multi-task SFT student on GPU 7 and returns; the runs keep going after logout. One queue, two
# jobs one after the other: Qwen3-0.6B on all of mt_v2 (EE v2 + NER v2 + TACRED + FewRel, 148,583 rows),
# then on the fixed square-root subset (SUBSET=0.5, 41,183 rows). CE only, no teacher: the baseline of the
# multi-teacher (BAM) students. Same setup as run.sh (venv, PET_* unset, port 29500 + 10 x GPU).
# Console output: logs/sft_student_0.6B_mt_v2[_subset0.5].log. Training log with the RESULT lines (one per
# dataset plus a "tasks" line with EE, NER, RE and their mean):
# results/qwen3/general_domain/sft_0.6B_mt_v2[_subset0.5]/<settings>/log.txt
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

# mt_v2 and the label maps: run.sh unpacked processed_data.zip before mt_v2 existed, so mt_v2 comes out of
# the zip here; a zip without it has to be fetched again with download.txt
for f in data/general_domain/ee_label_map.tsv data/general_domain/ner_label_map.tsv; do
    if [ ! -f ${f} ]; then
        echo "missing ${f}: run run.sh once to unpack processed_data.zip"
        exit 1
    fi
done
if [ ! -f processed_data/general_domain/mt_v2/qwen/train_0.bin ]; then
    python tools/unpack_data.py mt_v2 || exit 1
fi

GPU=7
PORT=$((29500 + 10 * GPU))
LOGS=(logs/sft_student_0.6B_mt_v2.log logs/sft_student_0.6B_mt_v2_subset0.5.log)
mark=logs/.run3_started
rm -f ${mark} "${LOGS[@]}"
nohup bash -c ": > ${mark}; \
    GPU=${GPU} PORT=${PORT} bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh > ${LOGS[0]} 2>&1; \
    GPU=${GPU} PORT=${PORT} SUBSET=0.5 bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh > ${LOGS[1]} 2>&1" \
    > /dev/null 2>&1 &
pid=$!
# wait until the queue's own bash runs: before that a hang-up from the runner would kill it (see run.sh)
for i in $(seq 50); do [ -f ${mark} ] && break; sleep 0.2; done
if [ -f ${mark} ]; then
    echo "started gpu${GPU} (pid ${pid}): mt_v2 all rows, then SUBSET=0.5"
else
    echo "gpu${GPU} queue did not start within 10 s"
fi

echo "follow: tail -f ${LOGS[*]}    results: grep -rh RESULT results/qwen3/general_domain/sft_0.6B_mt_v2* --include=log.txt"

# the logs also go to this console until the queue ends; if the runner stops this script, only the console
# copy stops, the training goes on
tail -n +1 -F --pid=${pid} "${LOGS[@]}" 2>/dev/null
echo "gpu${GPU} queue finished"
