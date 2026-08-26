#!/usr/bin/env bash

uv python install 3.10

export UV_PROJECT_ENVIRONMENT=vlm_distill
uv sync --locked

source vlm_distill/bin/activate

bash download_datatrain.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7
bash script_train/run_baseline.sh
