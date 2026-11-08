# CUDA_VISIBLE_DEVICES=0 bash llama2_contra_csd.sh > logs/llama2_contra_csd_retrain_velocity.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 bash llama2_contra_csd_2.sh > logs/llama2_contra_csd_distill_3_layers.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash llama2_contra_csd_3.sh > logs/llama2_contra_csd_mode_us.log 2>&1 &

wait
