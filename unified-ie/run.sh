#! /bin/bash
# Starts the four RE SFT runs at once, one GPU each, and returns. The runs keep going after logout.
#   GPU 4  teacher Qwen3-4B (LoRA)   TACRED
#   GPU 5  teacher Qwen3-4B (LoRA)   FewRel
#   GPU 6  student Qwen3-0.6B (full) TACRED
#   GPU 7  student Qwen3-0.6B (full) FewRel
# Console output: logs/<run>.log. Training log with the RESULT lines: results/qwen3/general_domain/sft_*/<settings>/log.txt
cd "$(dirname "$0")"
mkdir -p logs

# tokenized data + label maps: download.txt fetches processed_data.zip; unpack it once (Python's zipfile:
# the image has no unzip binary)
if [ ! -f processed_data/general_domain/re_tacred/qwen/train_0.bin ]; then
    if [ ! -f processed_data.zip ]; then
        echo "missing processed_data.zip: fetch it with download.txt first"
        exit 1
    fi
    echo "unpacking processed_data.zip"
    python -c "import zipfile; zipfile.ZipFile('processed_data.zip').extractall('.')" || exit 1
fi

for d in re_tacred re_fewrel; do
    if [ ! -f processed_data/general_domain/${d}/qwen/train_0.bin ]; then
        echo "missing processed_data/general_domain/${d}/qwen after unpacking processed_data.zip"
        exit 1
    fi
done

for run in sft_teacher_4B_re_tacred sft_teacher_4B_re_fewrel sft_student_0.6B_re_tacred sft_student_0.6B_re_fewrel; do
    nohup bash scripts/general_domain/re/${run}.sh > logs/${run}.log 2>&1 &
    echo "started ${run} (pid $!) -> logs/${run}.log"
done

echo "follow: tail -f logs/*.log    results: grep -rh RESULT results/qwen3/general_domain --include=log.txt"
