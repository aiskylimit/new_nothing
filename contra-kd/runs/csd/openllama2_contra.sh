BASE_PATH=${1-"."}
export HF_HOME="/mnt/data/huggingface"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
DEVICE=0 # set gpu id here, 1 gpu only

CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/openllama2/csd/contra_3B_7B_teacher_lora.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ${BASE_PATH} ${MASTER_PORT} 1 openllama2-3B contra/csd/3B_7B openlm-research/open_llama_3b_v2 --seed $seed --eval-batch-size 32
    done
done
