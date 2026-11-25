#d
#datasets
--hf-dataset TIGER-Lab/MMEB-train /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_train/MMEB-train
--hf-dataset VoCuc/vlm-teacher-embedding /mnt/local/@PROJECT@/talas_vlm_embed/datasets
--hf-dataset TIGER-Lab/MMEB-eval /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_eval/MMEB-eval
--url https://huggingface.co/datasets/VoCuc/vlm-teacher-embedding/resolve/main/gmm.tar.gz /mnt/local/@PROJECT@/talas_vlm_embed
#models
--hf raghavlite/B3_Qwen2_2B /mnt/local/@PROJECT@/talas_vlm_embed/models/B3_Qwen2_2B
--hf raghavlite/B3_Qwen2_7B /mnt/local/@PROJECT@/talas_vlm_embed/models/B3_Qwen2_7B
--hf apple/FastVLM-0.5B /mnt/local/@PROJECT@/talas_vlm_embed/models/FastVLM-0.5B
--hf llava-hf/llava-onevision-qwen2-0.5b-ov-hf /mnt/local/@PROJECT@/talas_vlm_embed/models/llava-onevision-qwen2-0.5b-ov-hf
--hf Qwen/Qwen2-VL-2B-Instruct /mnt/local/@PROJECT@/talas_vlm_embed/models/Qwen/Qwen2-VL-2B-Instruct
--hf Qwen/Qwen2-VL-7B-Instruct /mnt/local/@PROJECT@/talas_vlm_embed/models/Qwen/Qwen2-VL-7B-Instruct

#unified
#v3




#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Embed_Gvendi/MMEB-evaloutputs-json/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/SElOT/results/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/MRL-text/checkpoints/mrl/bge-m3/bs128_lr2e-5_t0.05_ep1_s42/mteb_epoch_1/ +a

# sleep 2
# kill -9 $(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./cred/multi_gpu_v2.py


# cd ./cred
# bash ./project_commands.sh

# cd ./multi-mode-distill
# bash ./eval_rivad.sh
# bash ./project_commands.sh


# cd ./VLM_Embed_Gvendi
# bash ./project_commands.sh

# cd ./unified-ie
# bash ./project_commands.sh

# cd VLM_Distillation-main
# bash ./script_train/benchmark_qwen2_teacher_7b_fastvlm_student_05b_4gpu.sh
# bash ./project_commands_eval_qwen25vl_7b_gpu7.sh
# bash ./project_commands.sh
# bash ./project_commands_collect_eval_summary.sh
# bash ./project_commands_eval_base_model_1.sh

# cd ./SElOT
# bash ./project_commands.sh

# cd ./MRL-text
# bash ./project_commands.sh