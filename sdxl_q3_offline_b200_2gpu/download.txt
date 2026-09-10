#d
#datasets
--hf-dataset liuhuohuo2/pick-a-pic-v2 /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/data/pickapic_v2_full
--hf-dataset nateraw/parti-prompts /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/eval_sources/parti-prompts
--hf-dataset ymhao/HPDv2 /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/eval_sources/HPDv2
#models
--hf stabilityai/stable-diffusion-xl-base-1.0 /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/models/stable-diffusion-xl-base-1.0
--hf madebyollin/sdxl-vae-fp16-fix /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/models/sdxl-vae-fp16-fix
--hf laion/CLIP-ViT-H-14-laion2B-s32B-b79K /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/CLIP-ViT-H-14-laion2B-s32B-b79K
--hf yuvalkirstain/PickScore_v1 /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/PickScore_v1
--hf bert-base-uncased /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/bert-base-uncased
--url https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/HPSv2/
--url https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/HPSv2/
--url https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/open_clip/
--url https://huggingface.co/trl-lib/ddpo-aesthetic-predictor/resolve/main/aesthetic-model.pth /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/ddpo-aesthetic-predictor/
--url https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/ImageReward/
--url https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json /mnt/local/aiskylimit_new_nothing/sdxl_q3_offline_b200_2gpu/offline_assets/reward_models/ImageReward/
