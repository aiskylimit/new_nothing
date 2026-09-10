#!/usr/bin/env bash
# Central configuration for the offline SDXL Q3/DSPO pipeline.
# Every value can be overridden before invoking project_command.sh.

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export ASSET_ROOT="${ASSET_ROOT:-$PROJECT_ROOT/offline_assets}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/runtime}"

export PIPELINE_MODE="${PIPELINE_MODE:-full851k}"
# The scheduler may allocate any two physical GPUs, but normally remaps the
# visible devices to logical IDs 0 and 1 inside the job. Override GPU_IDS only
# when the allocation exposes different logical IDs.
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export NUM_GPUS="${NUM_GPUS:-2}"
export TARGET_GPU_FAMILY="${TARGET_GPU_FAMILY:-B200}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-64}"
export SEED="${SEED:-42}"

export MODEL_DIR="${MODEL_DIR:-$ASSET_ROOT/models/stable-diffusion-xl-base-1.0}"
export VAE_DIR="${VAE_DIR:-$ASSET_ROOT/models/sdxl-vae-fp16-fix}"
export DATA_DIR="${DATA_DIR:-$ASSET_ROOT/data/pickapic_v2_full/data}"
export STREAM_MANIFEST="${STREAM_MANIFEST:-$DATA_DIR/manifest-full851k.json}"

export EVAL_SOURCE_DIR="${EVAL_SOURCE_DIR:-$ASSET_ROOT/eval_sources}"
export RATIO_PICKAPIC_PROMPTS_SOURCE="${RATIO_PICKAPIC_PROMPTS_SOURCE:-$ASSET_ROOT/data/pickapic_v2_full/data/test_unique-00000-of-00001-ecef20a3469ae6e6.parquet}"
export RATIO_PARTIPROMPT_SOURCE="${RATIO_PARTIPROMPT_SOURCE:-$EVAL_SOURCE_DIR/parti-prompts/PartiPrompts.tsv}"
export RATIO_HPDV2_SOURCE="${RATIO_HPDV2_SOURCE:-$EVAL_SOURCE_DIR/HPDv2/test.json}"

export REWARD_ROOT="${REWARD_ROOT:-$ASSET_ROOT/reward_models}"
export RATIO_PICKSCORE_PROCESSOR_DIR="${RATIO_PICKSCORE_PROCESSOR_DIR:-$REWARD_ROOT/CLIP-ViT-H-14-laion2B-s32B-b79K}"
export RATIO_PICKSCORE_MODEL_DIR="${RATIO_PICKSCORE_MODEL_DIR:-$REWARD_ROOT/PickScore_v1}"
export RATIO_HPSV2_CHECKPOINT="${RATIO_HPSV2_CHECKPOINT:-$REWARD_ROOT/HPSv2/HPS_v2.1_compressed.pt}"
export RATIO_HPSV2_BPE="${RATIO_HPSV2_BPE:-$REWARD_ROOT/HPSv2/bpe_simple_vocab_16e6.txt.gz}"
export RATIO_AESTHETIC_CLIP_CHECKPOINT="${RATIO_AESTHETIC_CLIP_CHECKPOINT:-$REWARD_ROOT/open_clip/ViT-L-14.pt}"
export RATIO_AESTHETIC_MODEL_CHECKPOINT="${RATIO_AESTHETIC_MODEL_CHECKPOINT:-$REWARD_ROOT/ddpo-aesthetic-predictor/aesthetic-model.pth}"
export RATIO_IMAGEREWARD_ROOT="${RATIO_IMAGEREWARD_ROOT:-$REWARD_ROOT/ImageReward}"
export RATIO_IMAGEREWARD_BERT_DIR="${RATIO_IMAGEREWARD_BERT_DIR:-$REWARD_ROOT/bert-base-uncased}"

export VENV_DIR="${VENV_DIR:-/mnt/local/uvenvs/sdxl-q3-offline-b200-2gpu}"
export RUNS_DIR="${RUNS_DIR:-$RUNTIME_ROOT/runs}"
export EVAL_ROOT_OVERRIDE="${EVAL_ROOT_OVERRIDE:-$RUNTIME_ROOT/eval}"
export CACHE_DIR="${CACHE_DIR:-$RUNTIME_ROOT/cache}"
export STREAM_CACHE="${STREAM_CACHE:-$RUNTIME_ROOT/stream-cache}"
export LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/logs}"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"

# Hard offline boundary. Missing assets must fail instead of falling back to Hub.
export RATIO_OFFLINE_STRICT=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export RATIO_MODEL_FAMILY=sdxl
export RATIO_MODEL_ID="$MODEL_DIR"
export RATIO_VAE_ID="$VAE_DIR"
export RATIO_IMAGE_RESOLUTION="${RATIO_IMAGE_RESOLUTION:-1024}"
export RATIO_GENERATION_BATCH_SIZE="${RATIO_GENERATION_BATCH_SIZE:-2}"
export PILOT_PROMPT_COUNT="${PILOT_PROMPT_COUNT:-4}"
export OFFLINE_EVAL_LIMIT="${OFFLINE_EVAL_LIMIT:-0}"
