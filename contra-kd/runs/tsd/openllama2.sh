BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# TSD-KD
bash ${BASE_PATH}/scripts/openllama2/tsd/tsd_3B_7B_teacher_lora.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B tsd-kd/3B_7B openlm-research/open_llama_3b_v2 --seed $seed --eval-batch-size 32
    done
done
