#! /bin/bash
# Starts SFT runs (teacher Qwen3-4B LoRA, student Qwen3-0.6B full) and returns; the runs keep going after
# logout. Teachers are trained one per source dataset (script:dataset passes the dataset to the script,
# which keeps only its rows); students one per task. One queue per GPU, so each GPU runs one job at a time.
# Now, on GPUs 4-6 (GPU 7 runs run2.sh): teachers go to 5 epochs. The 3 EE teachers whose 2026-10-26 run used
# the old EE data (no arguments) train again from scratch; the other 9 teachers go on from their 3-epoch
# checkpoint for 2 more (the 3 are done, so only those 9 are started now). The full plan of the first round is kept below, commented out.
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
start () {  # $1 = GPU, then jobs: script path under scripts/general_domain/, optionally :<dataset>, optionally
            # @<teacher checkpoint> to train that LoRA on for MORE_EPOCHS (log name gets _cont)
    local gpu=$1; shift
    local mark=logs/.gpu${gpu}_started
    local cmd=": > ${mark}; "
    for j in "$@"; do
        local ckpt="" env="GPU=${gpu} PORT=$((29500 + 10 * gpu))"
        if [ "${j}" != "${j%@*}" ]; then
            ckpt=${j#*@} j=${j%@*}
            env+=" PEFT_PATH=${ckpt} EPOCHS=${MORE_EPOCHS}"
        fi
        local s=${j%%:*} ds=""
        [ "${j}" != "${s}" ] && ds=${j#*:}
        cmd+="${env} bash scripts/general_domain/${s}.sh ${ds} > logs/$(basename ${s})${ds:+_${ds}}${ckpt:+_cont}.log 2>&1; "
    done
    rm -f ${mark}
    nohup bash -c "${cmd}" > /dev/null 2>&1 &
    local pid=$!
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

# the full plan (all 17 runs, queues balanced on train rows x epochs, the 4B teacher about 5x the student):
# start 4 ner/sft_teacher_4B_ner:ontonotes re/sft_student_0.6B_re_tacred
# start 5 ee/sft_teacher_4B_ee_v1:maven re/sft_teacher_4B_re_fewrel ee/sft_teacher_4B_ee_v1:rams
# start 6 ee/sft_teacher_4B_ee_v2:maven ner/sft_teacher_4B_ner:conll03 ner/sft_student_0.6B_ner
# start 7 ee/sft_teacher_4B_ee_v2:rams re/sft_teacher_4B_re_tacred \
#     ee/sft_teacher_4B_ee_v1:ace ee/sft_teacher_4B_ee_v2:ace ee/sft_teacher_4B_ee_v1:geneva ee/sft_teacher_4B_ee_v2:geneva \
#     ee/sft_student_0.6B_ee_v1 ee/sft_student_0.6B_ee_v2 re/sft_student_0.6B_re_fewrel

# rerun on the current EE data. The old-data runs save to the same folders, so they are moved aside first:
# otherwise the new checkpoints overwrite theirs step by step and log.txt mixes old and new RESULT lines.
for r in sft_4B_ee_v1_maven sft_4B_ee_v2_maven sft_4B_ee_v1_rams; do
    src=results/qwen3/general_domain/${r}
    if [ -d "${src}" ] && [ ! -e results/qwen3/general_domain/_old_ee_data/${r} ]; then
        mkdir -p results/qwen3/general_domain/_old_ee_data
        mv "${src}" results/qwen3/general_domain/_old_ee_data/ && echo "moved ${src} to results/qwen3/general_domain/_old_ee_data/"
    fi
done

# teachers now train 5 epochs (the scripts' default). Those 3 start from scratch; the other 9 trained 3 epochs
# on the current data and go on from their last checkpoint for 2 more, saved apart in
# <settings e2-...>-from<step>/ (fresh optimizer and lr schedule, epochs in its log count from 0 again).
MORE_EPOCHS=2
cont () {  # $1 = job, $2 = run folder: "$1@<highest step of its 3-epoch run>", nothing if there is none
    local c
    c=$(ls -d results/qwen3/general_domain/$2/e3-*/[0-9]* 2>/dev/null | sort -t/ -k6,6n | tail -1)
    if [ -z "${c}" ]; then echo "no 3-epoch checkpoint for $2: not continued" >&2; return; fi
    echo "$1@${c}"
}
# first try (2026-10-27): the 3 from scratch finished; the 9 continued ones all stopped at argparse (the
# suffix "-from<step>" was read as a flag), fixed in the teacher scripts.
# start 4 ee/sft_teacher_4B_ee_v1:maven \
#     $(cont re/sft_teacher_4B_re_tacred sft_4B_re_tacred) \
#     $(cont ee/sft_teacher_4B_ee_v1:ace sft_4B_ee_v1_ace) $(cont ee/sft_teacher_4B_ee_v2:ace sft_4B_ee_v2_ace)
# start 5 ee/sft_teacher_4B_ee_v2:maven \
#     $(cont re/sft_teacher_4B_re_fewrel sft_4B_re_fewrel)
# start 6 ee/sft_teacher_4B_ee_v1:rams \
#     $(cont ner/sft_teacher_4B_ner:ontonotes sft_4B_ner_v2_ontonotes) $(cont ner/sft_teacher_4B_ner:conll03 sft_4B_ner_v2_conll03) \
#     $(cont ee/sft_teacher_4B_ee_v1:geneva sft_4B_ee_v1_geneva) $(cont ee/sft_teacher_4B_ee_v2:geneva sft_4B_ee_v2_geneva) \
#     $(cont ee/sft_teacher_4B_ee_v2:rams sft_4B_ee_v2_rams)
# now only the 9 continued ones, queues balanced on 2/3 of the 3-epoch time (OntoNotes ~105 min, FewRel ~65,
# CoNLL03 ~35, ...), about 1.5-1.8 h each
start 4 $(cont ner/sft_teacher_4B_ner:ontonotes sft_4B_ner_v2_ontonotes)
start 5 $(cont re/sft_teacher_4B_re_fewrel sft_4B_re_fewrel) $(cont ee/sft_teacher_4B_ee_v2:rams sft_4B_ee_v2_rams) \
    $(cont ee/sft_teacher_4B_ee_v1:ace sft_4B_ee_v1_ace) $(cont ee/sft_teacher_4B_ee_v2:ace sft_4B_ee_v2_ace)
start 6 $(cont ner/sft_teacher_4B_ner:conll03 sft_4B_ner_v2_conll03) $(cont re/sft_teacher_4B_re_tacred sft_4B_re_tacred) \
    $(cont ee/sft_teacher_4B_ee_v1:geneva sft_4B_ee_v1_geneva) $(cont ee/sft_teacher_4B_ee_v2:geneva sft_4B_ee_v2_geneva)

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"
