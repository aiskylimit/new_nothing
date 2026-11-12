#d
#datasets
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/llava_v1_5_mix665k.json /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/train_val_images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/textvqa
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/coco/train2017.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/coco
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/gqa/images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/gqa
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/ocr_vqa/ocr_vqa_images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/ocr_vqa
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/ocr_vqa/dataset.json /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/ocr_vqa
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/vg/images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/vg
--hf-dataset https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/vg/images2.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/vg
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/GQA_TestDev_Balanced.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MME.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/RealWorldQA.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/ScienceQA_TEST.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/AI2D_TEST_NO_MASK.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MMMU_DEV_VAL.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MMStar.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/ChartQA_TEST.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/DocVQA_VAL.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/TextVQA_VAL.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData
--hf-dataset https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/OCRBench.tsv /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/eval_data/LMUData

#log
#v1


#2 -f-/mnt/local/aiskylimit_new_nothing/cred/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/unified-ie/logs/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/eval-impact/outputs/llama2-tinyllama-opd/logs/dolly-dskd-opd-10.log

# sleep 2
# kill -9 $(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader)
# kill -9 117225
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