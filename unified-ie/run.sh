#! /bin/bash
# Starts every SFT run (teacher Qwen3-4B LoRA, student Qwen3-0.6B full) and returns; the runs keep going
# after logout. One queue per GPU, so each GPU runs one job at a time, in this order:
#   GPU 4  teacher NER
#   GPU 5  teacher EE v1, teacher FewRel
#   GPU 6  teacher EAE v1, teacher EE v2, student NER
#   GPU 7  teacher TACRED, student TACRED, teacher EAE v2, student EAE v1, student EAE v2,
#          student EE v1, student EE v2, student FewRel
# The queues are balanced on rough cost (train rows x epochs, the 4B teacher about 5x the student).
# EE = event detection on ACE/MAVEN/RAMS/GENEVA (ee_v1, ee_v2), EAE = events with arguments on ACE/GENEVA
# (ee_arg_v1, ee_arg_v2), NER = CoNLL-2003 + OntoNotes (ner_v2), RE = TACRED, FewRel.
# Console output: logs/<run>.log. Training log with the RESULT lines: results/qwen3/general_domain/sft_*/<settings>/log.txt
cd "$(dirname "$0")"
mkdir -p logs

# tokenized data + label maps: download.txt fetches processed_data.zip; unpack it once (Python's zipfile:
# the image has no unzip binary)
if [ ! -f processed_data/general_domain/ner_v2/qwen/train_0.bin ]; then
    if [ ! -f processed_data.zip ]; then
        echo "missing processed_data.zip: fetch it with download.txt first"
        exit 1
    fi
    echo "unpacking processed_data.zip"
    python -c "import zipfile; zipfile.ZipFile('processed_data.zip').extractall('.')" || exit 1
fi

for d in ee_v1 ee_v2 ee_arg_v1 ee_arg_v2 re_tacred re_fewrel ner_v2; do
    if [ ! -f processed_data/general_domain/${d}/qwen/train_0.bin ]; then
        echo "missing processed_data/general_domain/${d}/qwen after unpacking processed_data.zip"
        exit 1
    fi
done

# one queue per GPU: the scripts in a queue run one after the other
start () {  # $1 = queue name, then scripts (paths under scripts/general_domain/)
    local name=$1; shift
    local cmd=""
    for s in "$@"; do
        cmd+="bash scripts/general_domain/${s}.sh > logs/$(basename ${s}).log 2>&1; "
    done
    nohup bash -c "${cmd}" > /dev/null 2>&1 &
    echo "started ${name} (pid $!): $*"
}

start gpu4 ner/sft_teacher_4B_ner
start gpu5 ee/sft_teacher_4B_ee_v1 re/sft_teacher_4B_re_fewrel
start gpu6 ee/sft_teacher_4B_eae_v1 ee/sft_teacher_4B_ee_v2 ner/sft_student_0.6B_ner
start gpu7 re/sft_teacher_4B_re_tacred re/sft_student_0.6B_re_tacred ee/sft_teacher_4B_eae_v2     ee/sft_student_0.6B_eae_v1 ee/sft_student_0.6B_eae_v2 ee/sft_student_0.6B_ee_v1 ee/sft_student_0.6B_ee_v2     re/sft_student_0.6B_re_fewrel

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"
