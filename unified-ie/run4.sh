#! /bin/bash
# The multi-task students on mt_v2 (EE v2 + NER v2 + TACRED + FewRel), once the 8 teachers are trained:
# 2 SFT students (CE only, the baseline) and 4 multi-teacher BAM students, each on all train rows and on the
# fixed square-root subset (SUBSET=0.5). Starts one queue per GPU 4-7 and returns; the runs keep going after
# logout. Same setup as run.sh (venv, PET_* unset, port 29500 + 10 x GPU).
#   GPU 4  BAM, KD on the logits only (fkl, teacher annealing), all rows                        ~longest
#   GPU 5  BAM + EventKD losses (sfkl + span loss, spans where the answer writes them), all rows
#   GPU 6  BAM, then BAM + EventKD, both on the square-root subset (41,183 rows)
#   GPU 7  SFT student on all rows, then on the square-root subset
# Each BAM student learns from the teacher of its row's dataset: the best dev checkpoint of
# results/qwen3/general_domain/sft_4B_{ee_v2_ace,ee_v2_geneva,ee_v2_maven,ee_v2_rams,ner_v2_conll03,
# ner_v2_ontonotes,re_tacred,re_fewrel} (tools/best_lora.py, -from<step> continuations included).
# Console output: logs/<run>.log. Training logs with the RESULT lines (one per dataset plus a "tasks" line
# with EE, NER, RE and their mean): results/qwen3/general_domain/{sft_0.6B_mt_v2*,bam_0.6B_4B_mt_v2}/.../log.txt
# SMOKE=1 bash run4.sh: every job trains 2 optimizer steps on 64 rows, saves, scores 16 dev and 16 test rows,
# into results_smoke/ and logs/smoke_*.log on port +1, to catch a missing teacher or an OOM in a few minutes.
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

# the 8 teachers: stop now rather than in 4 queues if one has no finished epoch
echo "teachers (best dev checkpoint):"
for r in ee_v2_ace ee_v2_geneva ee_v2_maven ee_v2_rams ner_v2_conll03 ner_v2_ontonotes re_tacred re_fewrel; do
    lora=$(python tools/best_lora.py results/qwen3/general_domain/sft_4B_${r}) || exit 1
    echo "  ${r}: ${lora}"
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

SFT="bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh"
BAM="bash scripts/general_domain/bam_0.6B_4B.sh mt_v2"
EVENTKD="KD_TYPE=sfkl W_SPAN=1.0"

QUEUE_LOGS=() QUEUE_PIDS=()
start () {  # $1 = GPU, then pairs: <log name> "<env and command>", run one after the other on that GPU
    local gpu=$1; shift
    local mark=logs/.${PREFIX}run4_gpu${gpu}_started
    local cmd=": > ${mark}; " names=""
    while [ $# -gt 0 ]; do
        local log=logs/${PREFIX}$1.log
        cmd+="GPU=${gpu} PORT=$((29500 + 10 * gpu + PORT_OFFSET)) $2 > ${log} 2>&1; "
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

start 4 bam_mt_v2 "${BAM}"
start 5 bam_eventkd_mt_v2 "${EVENTKD} ${BAM}"
start 6 bam_mt_v2_subset0.5 "SUBSET=0.5 ${BAM}" \
        bam_eventkd_mt_v2_subset0.5 "SUBSET=0.5 ${EVENTKD} ${BAM}"
start 7 sft_student_0.6B_mt_v2 "${SFT}" \
        sft_student_0.6B_mt_v2_subset0.5 "SUBSET=0.5 ${SFT}"

echo "follow: tail -f ${QUEUE_LOGS[*]}"
echo "results: grep -rh 'RESULT test' results/qwen3/general_domain/sft_0.6B_mt_v2* results/qwen3/general_domain/bam_0.6B_4B_mt_v2 --include=log.txt"

# the logs also go to this console until every queue ends; if the runner stops this script, only the
# console copy stops, the training goes on
tail -n +1 -F "${QUEUE_LOGS[@]}" 2>/dev/null &
TAIL=$!
alive () { for p in "${QUEUE_PIDS[@]}"; do kill -0 ${p} 2>/dev/null && return 0; done; return 1; }
while alive; do sleep 60; done
kill ${TAIL} 2>/dev/null
echo "run4 queues finished"
