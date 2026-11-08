CUDA_VISIBLE_DEVICES=0 bash runs/tsd/gpt2.sh > logs/gpt2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash runs/tsd/gpt2_contra.sh > logs/gpt2_contra.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash runs/tsd/openllama2.sh > logs/openllama2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash runs/tsd/openllama2_contra.sh > logs/openllama2_contra.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 bash runs/tsd/llama2.sh > logs/llama2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 bash runs/tsd/llama2_contra.sh > logs/llama2_contra.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 bash runs/csd/llama2_contra.sh > logs/llama2_contra_csd.log 2>&1 &

wait
