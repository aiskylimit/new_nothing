BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# TSD-KD
bash ${BASE_PATH}/scripts/gpt2/tsd/tsd_base.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 tsd-kd/base --seed $seed --eval-batch-size 64
    done
done
