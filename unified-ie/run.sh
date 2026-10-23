#! /bin/bash
# Starts the SFT runs, one GPU each, and returns. The runs keep going after logout.
#   GPU 4  teacher Qwen3-4B (LoRA)   TACRED, then NER (CoNLL-2003 + OntoNotes)
#   GPU 5  teacher Qwen3-4B (LoRA)   FewRel
#   GPU 6  student Qwen3-0.6B (full) TACRED, then NER
#   GPU 7  student Qwen3-0.6B (full) FewRel
# TACRED (4.6k train rows) finishes long before FewRel (32k), so NER follows it on the same GPU.
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

for d in re_tacred re_fewrel ner_v2; do
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

start gpu4 re/sft_teacher_4B_re_tacred ner/sft_teacher_4B_ner
start gpu5 re/sft_teacher_4B_re_fewrel
start gpu6 re/sft_student_0.6B_re_tacred ner/sft_student_0.6B_ner
start gpu7 re/sft_student_0.6B_re_fewrel

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"
