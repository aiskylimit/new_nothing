BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# TSD-KD + Contra
# stage 1
# bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} 2012 1
# stage 2
bash ${BASE_PATH}/scripts/gpt2/tsd/contra_base.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 contra/tsd-kd/base --seed $seed --eval-batch-size 64
    done
done
