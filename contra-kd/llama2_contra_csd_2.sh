BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
# DEVICE=0 # set gpu id here, 1 gpu only

bash ${BASE_PATH}/scripts/llama2/train_velocity_field.sh ${BASE_PATH} ${MASTER_PORT} 1 \
    --num-distill-layers 3 \
    --save "${BASE_PATH}/results/llama2/train/velocity_field_1"

bash ${BASE_PATH}/scripts/llama2/csd/contra_7B_13B_teacher_lora.sh ${BASE_PATH} ${MASTER_PORT} 1 \
    --eval-batch-size 16 --batch-size 8 --gradient-accumulation-steps 2 \
    --velocity-field-path "${BASE_PATH}/results/llama2/train/velocity_field_1/velocity_field.pth" \
    --projector-path "${BASE_PATH}/results/llama2/train/velocity_field_1/projector.pth" \
    --num-distill-layers 3 \
    --save "${BASE_PATH}/results/llama2/train/contra/csd/3layers"

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ${BASE_PATH}/scripts/llama2/eval/eval_main_${benchmark}_lora.sh ${BASE_PATH} ${MASTER_PORT} 1 llama2-7B contra/csd/3layers meta-llama/Llama-2-7b-hf --seed $seed --eval-batch-size 32
    done
done
