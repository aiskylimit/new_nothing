#! /bin/bash
# Starts every SFT run (teacher Qwen3-4B LoRA, student Qwen3-0.6B full) and returns; the runs keep going
# after logout. Teachers are trained one per source dataset (script:dataset passes the dataset to the
# script, which keeps only its rows); students one per task. One queue per GPU, so each GPU runs one job
# at a time, teachers first. The queues are balanced on rough cost (train rows x epochs, the 4B teacher
# about 5x the student), about 850k-980k row-epoch units each.
# EE = events with arguments on ACE/MAVEN/RAMS/GENEVA (ee_v1, ee_v2; MAVEN/RAMS have no arguments, so the
# loss skips their []), NER = CoNLL-2003 + OntoNotes (ner_v2), RE = TACRED, FewRel.
# Console output: logs/<script>[_<dataset>].log. Training log with the RESULT lines:
# results/qwen3/general_domain/sft_*/<settings>/log.txt (teachers: sft_4B_<data>_<dataset>)
cd "$(dirname "$0")"
mkdir -p logs

# every run needs the unified-ie venv (peft etc.); the image's own python3.10 has no peft. The queues
# below are started from this shell, so they inherit it.
VENV=/mnt/local/uvenvs/unified-ie
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: ${VENV} does not exist" >&2
    exit 1
fi
source "${VENV}/bin/activate"

# the pod sets PET_* rendezvous variables; torchrun reads them over --master_port, so every run would join
# one shared rendezvous and die with whichever run hosts it (same fix as OpenED/project_commands_lib.sh)
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done

# tokenized data + label maps: download.txt fetches processed_data.zip; unpack it once (Python's zipfile:
# the image has no unzip binary)
# The EE rows before 2026-10-26 had no arguments; the current ones mark MAVEN/RAMS with "unknown_args". A
# processed_data/ without it is unpacked again from the zip, and an old zip stops the run.
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

for d in ee_v1 ee_v2 re_tacred re_fewrel ner_v2; do
    if [ ! -f processed_data/general_domain/${d}/qwen/train_0.bin ]; then
        echo "missing processed_data/general_domain/${d}/qwen after unpacking processed_data.zip"
        exit 1
    fi
done

# one queue per GPU: the jobs in a queue run one after the other on that GPU, all on torchrun port
# 29500 + 10 x GPU (one job at a time per queue, so no two live runs share a port)
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

start 4 ner/sft_teacher_4B_ner:ontonotes re/sft_student_0.6B_re_tacred
start 5 ee/sft_teacher_4B_ee_v1:maven re/sft_teacher_4B_re_fewrel ee/sft_teacher_4B_ee_v1:rams
start 6 ee/sft_teacher_4B_ee_v2:maven ner/sft_teacher_4B_ner:conll03 ner/sft_student_0.6B_ner
start 7 ee/sft_teacher_4B_ee_v2:rams re/sft_teacher_4B_re_tacred \
    ee/sft_teacher_4B_ee_v1:ace ee/sft_teacher_4B_ee_v2:ace ee/sft_teacher_4B_ee_v1:geneva ee/sft_teacher_4B_ee_v2:geneva \
    ee/sft_student_0.6B_ee_v1 ee/sft_student_0.6B_ee_v2 re/sft_student_0.6B_re_fewrel

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"
