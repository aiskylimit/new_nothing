#d
#datasets
--hf-dataset TIGER-Lab/MMEB-train /mnt/local/@PROJECT@/VLM_Embed_Gvendi/vlm2vec_train/MMEB-train
--hf-dataset TIGER-Lab/MMEB-eval /mnt/local/@PROJECT@/VLM_Embed_Gvendi/vlm2vec_eval/MMEB-eval
--url https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip /mnt/local/@PROJECT@/VLM_Embed_Gvendi/vlm2vec_external/flickr30k
--url https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv /mnt/local/@PROJECT@/VLM_Embed_Gvendi/vlm2vec_external/flickr30k
--hf-dataset minhhotboy9x/VLM_embed_cache /mnt/local/@PROJECT@/VLM_Embed_Gvendi/datasets
#models
--hf raghavlite/B3_Qwen2_2B /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/B3_Qwen2_2B
--hf raghavlite/B3_Qwen2_7B /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/B3_Qwen2_7B
--hf apple/FastVLM-0.5B /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/FastVLM-0.5B
--hf llava-hf/llava-onevision-qwen2-0.5b-ov-hf /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/llava-onevision-qwen2-0.5b-ov-hf
--hf Qwen/Qwen2-VL-2B-Instruct /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/Qwen2-VL-2B-Instruct
--hf Qwen/Qwen2-VL-7B-Instruct /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/Qwen2-VL-7B-Instruct
--hf Qwen/Qwen2.5-VL-3B-Instruct /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/Qwen2.5-VL-3B-Instruct
--hf Qwen/Qwen3-VL-Embedding-8B /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/Qwen3-VL-Embedding-8B
--hf OpenGVLab/InternVL-14B-Flickr30K-FT-364px /mnt/local/@PROJECT@/VLM_Embed_Gvendi/models/InternVL-14B-Flickr30K-FT-364px

#log 
#v2


#2 -f-/mnt/local/aiskylimit_new_nothing/cred/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/eval-impact/outputs/llama2-tinyllama-opd/logs/dolly-dskd-opd-10.log

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
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./eval_rivad.sh
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh

# cd ./VLM_Embed_Gvendi
# bash ./project_commands_2.sh
# bash ./project_commands.sh

# cd ./unified-ie
# bash ./project_commands.sh

# cd ./eval-impact
# tree outputs
# CUDA_VISIBLE_DEVICES=2 bash ./run_8k_b200.sh


# cd VLM_Distillation-main
# bash ./project_commands.sh
# bash ./project_commands_collect_eval_summary.sh
# bash ./project_commands_eval_base_model_1.sh