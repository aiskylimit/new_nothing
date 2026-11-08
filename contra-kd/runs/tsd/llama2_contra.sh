BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# TSD-KD + Contra
# stage 1
# bash ${BASE_PATH}/scripts/llama2/train_velocity_field.sh ${BASE_PATH} 2012 1
# stage 2
bash ${BASE_PATH}/scripts/llama2/tsd/contra_7B_13B_teacher_lora.sh ${BASE_PATH} ${MASTER_PORT} 1 --eval-batch-size 16 --batch-size 4 --gradient-accumulation-steps 4

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ./scripts/llama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 llama2-7B contra/tsd-kd/7B_13B meta-llama/Llama-2-7b-hf --seed $seed --eval-batch-size 32
    done
done
