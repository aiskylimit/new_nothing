#!/usr/bin/env bash

uv python install 3.10

export UV_PROJECT_ENVIRONMENT=vlm_distill
uv sync --locked

source vlm_distill/bin/activate

unzip -q -o train_data/coco/train2017.zip -d coco
unzip -q -o train_data/gqa/images.zip -d gqa
unzip -q -o train_data/textvqa/train_val_images.zip -d textvqa
unzip -q -o train_data/vg/images.zip -d vg
unzip -q -o train_data/vg/images2.zip -d vg

#bash download_datatrain.sh

export CUDA_VISIBLE_DEVICES=4,5
bash script_train/run_baseline.sh
