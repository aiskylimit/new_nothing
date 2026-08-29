#d
#datasets
--url https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/llava_v1_5_mix665k.json
--url https://huggingface.co/datasets/DVLe/ocr_vqa/resolve/main/dataset.json /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/ocr_vqa/dataset.json
--url http://images.cocodataset.org/zips/train2017.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/coco/train2017.zip
--url https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/gqa/images.zip
--url https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/textvqa/train_val_images.zip
--url https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/vg/images.zip
--url https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip /mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/train_data/vg/images2.zip
#models
--hf Qwen/Qwen2.5-VL-3B-Instruct ./VLM_Distillation-main/models/Qwen/Qwen2.5-VL-3B-Instruct
--hf Qwen/Qwen3-VL-8B-Instruct ./VLM_Distillation-main/models/Qwen/Qwen3-VL-8B-Instruct
