BASE_PATH=${1-"."}
MASTER_PORT=2040
DEVICE=0
EVAL_BATCH_SIZE=64

# sft
bash ${BASE_PATH}/scripts/gpt2/sft/sft_base.sh ${BASE_PATH} 2012 1
# kd
bash ${BASE_PATH}/scripts/gpt2/kd/kd_base.sh ${BASE_PATH} 2012 1

# eval
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 sft/gpt2-base --seed $seed --eval-batch-size ${EVAL_BATCH_SIZE}
    done
done

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 kd/base --seed $seed --eval-batch-size ${EVAL_BATCH_SIZE}
    done
done
