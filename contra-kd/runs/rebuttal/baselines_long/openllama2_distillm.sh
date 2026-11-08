BASE_PATH=${1-"."}
MASTER_PORT=2040
DEVICE=0
EVAL_BATCH_SIZE=32

# distillm
bash ${BASE_PATH}/scripts/openllama2/distillm/train_3B_7B_teacher_lora.sh ${BASE_PATH} 2012 1

# eval
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B distillm/3B_7B openlm-research/open_llama_3b_v2 --seed $seed --eval-batch-size ${EVAL_BATCH_SIZE}
    done
done
