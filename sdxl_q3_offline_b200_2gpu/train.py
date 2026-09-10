#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,     StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

from ratio_diffusion import (
    StepConfidenceHead,
    auxiliary_output_gradient_scale,
    ddpm_posterior_coefficients,
    dspo_winner_score_anchor,
    diffusion_dpo_loss_from_per_sample_losses,
    equal_covariance_gaussian_kl,
    expected_transition_log_uplift,
    loss_output_gradient,
    latent_teacher_preference_probability,
    mode_seeking_exponent_clip_fraction,
    mode_seeking_margin_gradient_magnitude,
    mode_seeking_power_loss_from_log_ratio,
    preference_loss_margin_scale,
    predict_clean_from_epsilon,
    reduce_latent,
    sample_true_posterior_action,
    scaled_basu_exponent_clip_fraction,
    scaled_basu_loss_from_log_ratio,
    step_aware_preference_loss,
    step_preference_consistency_target,
    select_timestep_pair_scores,
    symmetric_reference_kl_anchor,
    transition_log_uplift,
    transition_mean,
    true_posterior_mean,
    winner_denoising_anchor,
)


if is_wandb_available():
    import wandb

    
    
## SDXL
import functools
import gc
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
}

BREGMAN_PREFERENCE_LOSSES = frozenset({"ratio_bregman", "step_aware_tbpo"})
MODE_SEEKING_PREFERENCE_LOSSES = frozenset(
    {"mode_seeking_ratio", "step_aware_mode_seeking_tbpo"}
)
STEP_AWARE_PREFERENCE_LOSSES = frozenset(
    {"step_aware_tbpo", "step_aware_mode_seeking_tbpo"}
)
RATIO_PREFERENCE_LOSSES = frozenset(
    {*BREGMAN_PREFERENCE_LOSSES, *MODE_SEEKING_PREFERENCE_LOSSES}
)

        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pickapic_streaming_manifest",
        type=str,
        default=None,
        help="Use the bounded-cache rank-partitioned Pick-a-Pic parquet streaming loader.",
    )
    parser.add_argument(
        "--pickapic_stream_cache",
        type=str,
        default=None,
        help="Small persistent cache used by the Pick-a-Pic streaming loader.",
    )
    parser.add_argument(
        "--streaming_retries",
        type=int,
        default=40,
        help="Download retries per remote parquet shard in streaming mode.",
    )
    parser.add_argument(
        "--streaming_prefetch_next_shard",
        action="store_true",
        help=(
            "Download the next shard while consuming the current one. Disabled by default "
            "to reduce concurrent Hugging Face range requests and bounded-cache pressure."
        ),
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Keep only this many resumable checkpoints; final pipeline files are unaffected.",
    )
    parser.add_argument(
        "--eval_checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Optional directory that retains an inference-ready UNet snapshot for every "
            "scheduled checkpoint. Resumable optimizer checkpoints can still be bounded "
            "with --checkpoints_total_limit."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--preference_loss",
        type=str,
        default="dpo",
        choices=[
            "dpo",
            "ratio_bregman",
            "mode_seeking_ratio",
            "step_aware_tbpo",
            "step_aware_mode_seeking_tbpo",
        ],
        help=(
            "Pairwise preference loss: original Diffusion-DPO, the existing Scaled-Basu "
            "Ratio Diffusion objective, theorem-preserving mode-seeking Ratio Diffusion, "
            "step-aware Scaled-Basu TBPO, or step-aware mode-seeking Power-Bregman TBPO."
        ),
    )
    parser.add_argument(
        "--ratio_beta",
        type=float,
        default=5000.0,
        help="Exponent/temperature for the transition-level model density ratio.",
    )
    parser.add_argument(
        "--ratio_margin_gradient_scale",
        type=float,
        default=None,
        help=(
            "Optional target magnitude gamma_R of dL/d(raw transition margin) at margin zero. "
            "When set, a constant loss multiplier decouples this update scale from --ratio_beta; "
            "the default None preserves the legacy objective exactly."
        ),
    )
    parser.add_argument(
        "--mode_seeking_kappa",
        type=float,
        default=0.5,
        help=(
            "Power-Bregman mode-seeking strength for --preference_loss=mode_seeking_ratio; "
            "must lie strictly between 0 and 1."
        ),
    )
    parser.add_argument(
        "--bregman_lambda",
        type=float,
        default=0.0,
        help="Scaled-Basu power parameter for ratio_bregman and step_aware_tbpo.",
    )
    parser.add_argument(
        "--bregman_scale",
        type=float,
        default=4.0,
        help="Positive global scale c_B for the Scaled-Basu loss.",
    )
    parser.add_argument(
        "--ratio_reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help=(
            "Latent reduction: practical dimension-normalized mean or theorem-literal sum."
        ),
    )
    parser.add_argument(
        "--transition_estimator",
        type=str,
        default=None,
        choices=["pointwise", "expected"],
        help=(
            "Transition estimator. Defaults to expected for mode_seeking_ratio (the v0.5 "
            "algorithm) and pointwise otherwise."
        ),
    )
    parser.add_argument(
        "--state_correction",
        type=str,
        default="exact_gaussian",
        choices=["exact_gaussian", "none"],
        help="TBPO-A state-only correction: exact fixed-variance Gaussian KL or no correction.",
    )
    parser.add_argument(
        "--detach_state_correction",
        action="store_true",
        help="Detach the plug-in Gaussian state correction (ablation; gradients flow by default).",
    )
    parser.add_argument(
        "--reference_anchor_output_gradient_ratio",
        type=float,
        default=0.0,
        help=(
            "Target output-gradient norm of the explicit symmetric fixed-reference KL anchor "
            "relative to the TBPO policy term. Zero disables the anchor."
        ),
    )
    parser.add_argument(
        "--winner_anchor_output_gradient_ratio",
        type=float,
        default=0.0,
        help=(
            "Target output-gradient norm of the selected preferred-image anchor relative "
            "to the TBPO policy term. Zero disables the anchor."
        ),
    )
    parser.add_argument(
        "--winner_anchor_type",
        type=str,
        default="mse",
        choices=["mse", "dspo_score"],
        help=(
            "Preferred-branch auxiliary: standard denoising MSE or a DSPO-shaped "
            "score target. This only has an effect when the winner anchor gradient ratio is positive."
        ),
    )
    parser.add_argument(
        "--dspo_probability_temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature used to convert the selected detached DSPO logit into the "
            "preferred probability for --winner_anchor_type=dspo_score."
        ),
    )
    parser.add_argument(
        "--dspo_probability_source",
        type=str,
        default="mse_preference",
        choices=["mse_preference", "tbpo_margin"],
        help=(
            "Detached probability used by the DSPO-shaped anchor. mse_preference follows "
            "the official SD1.5 DSPO recipe; tbpo_margin is an explicitly non-faithful ablation."
        ),
    )
    parser.add_argument(
        "--dspo_logit_beta",
        type=float,
        default=0.01,
        help=(
            "Positive beta_D multiplying the model-minus-reference winner/loser MSE "
            "difference when --dspo_probability_source=mse_preference. The official "
            "DSPO SD1.5 recipe recommends sweeping 0.001, 0.01, and 0.05."
        ),
    )
    parser.add_argument(
        "--dspo_score_correction_scale",
        type=float,
        default=0.25,
        help=(
            "Mu in [0, 1) multiplying the actionable current-minus-reference score "
            "correction in the DSPO-shaped winner anchor. Values below one prevent "
            "the detached residual Jacobian from vanishing or reversing."
        ),
    )
    parser.add_argument(
        "--auxiliary_output_gradient_scale_max",
        type=float,
        default=1000.0,
        help="Safety cap on either dynamically calibrated auxiliary-loss multiplier.",
    )
    parser.add_argument(
        "--max_exp_argument",
        type=float,
        default=30.0,
        help="Numerical safety bound applied only to exponential arguments in the Bregman loss.",
    )
    parser.add_argument(
        "--transition_variance_floor",
        type=float,
        default=1e-12,
        help="Positive floor for DDPM posterior variance in Ratio Diffusion.",
    )
    parser.add_argument(
        "--confidence_hidden_dim",
        type=int,
        default=128,
        help="Hidden width of the step-aware confidence MLP.",
    )
    parser.add_argument(
        "--confidence_learning_rate",
        type=float,
        default=1e-4,
        help=(
            "Learning rate for the confidence MLP; kept separate from the much "
            "smaller diffusion-backbone learning rate."
        ),
    )
    parser.add_argument(
        "--confidence_timestep_embedding_dim",
        type=int,
        default=32,
        help="Even sinusoidal timestep-embedding width for step-aware TBPO.",
    )
    parser.add_argument(
        "--confidence_loss_weight",
        type=float,
        default=1.0,
        help="Lambda multiplying confidence BCE in step-aware TBPO.",
    )
    parser.add_argument(
        "--confidence_min_policy_weight",
        type=float,
        default=0.05,
        help="Strictly positive floor for learned step weights.",
    )
    parser.add_argument(
        "--confidence_policy_weight_normalization",
        type=str,
        default="none",
        choices=["none", "global_microbatch_mean"],
        help=(
            "Normalize detached policy weights by their cross-rank microbatch mean. "
            "The default none preserves legacy absolute confidence gating."
        ),
    )
    parser.add_argument(
        "--confidence_label_smoothing",
        type=float,
        default=0.05,
        help="Symmetric smoothing applied to binary local-consistency targets.",
    )
    parser.add_argument(
        "--confidence_target_source",
        type=str,
        default="reference_mse",
        choices=["reference_mse", "precomputed_lrm"],
        help=(
            "Teacher for the step-aware confidence head: frozen-reference denoising "
            "consistency or precomputed noise-aware latent reward model (LRM) scores."
        ),
    )
    parser.add_argument(
        "--confidence_policy_signal",
        type=str,
        default="head",
        choices=["head", "lrm_teacher"],
        help=(
            "Source of detached TBPO policy weights. lrm_teacher directly uses the "
            "precomputed LRM preference probability while still distilling the confidence head."
        ),
    )
    parser.add_argument(
        "--lrm_score_0_column",
        type=str,
        default="lrm_score_0",
        help="Dataset column containing scalar or timestep-binned frozen-LRM scores for jpg_0.",
    )
    parser.add_argument(
        "--lrm_score_1_column",
        type=str,
        default="lrm_score_1",
        help="Dataset column containing scalar or timestep-binned frozen-LRM scores for jpg_1.",
    )
    parser.add_argument(
        "--lrm_score_temperature",
        type=float,
        default=1.0,
        help="Positive Bradley--Terry temperature for precomputed frozen-LRM score gaps.",
    )
    parser.add_argument(
        "--sync_pair_augmentations",
        action="store_true",
        help="Reserved for a future synchronized image-augmentation implementation; off by default.",
    )
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    parser.add_argument(
        "--pair_label_policy",
        type=str,
        default="legacy",
        choices=["legacy", "filter", "error"],
        help=(
            "Handling of non-binary human pair labels when no choice model is used: legacy keeps "
            "the historical behavior, filter removes ties before training, and error fails fast."
        ),
    )
    parser.add_argument(
        "--dreamlike_pairs_only", action="store_true", help="Only train on pairs where both generations are from dreamlike"
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max_train_samples must be positive when set.")
    if (
        not math.isfinite(args.proportion_empty_prompts)
        or not 0.0 <= args.proportion_empty_prompts <= 1.0
    ):
        raise ValueError("--proportion_empty_prompts must lie in [0, 1].")
    for argument_name, value in (
        (
            "--reference_anchor_output_gradient_ratio",
            args.reference_anchor_output_gradient_ratio,
        ),
        (
            "--winner_anchor_output_gradient_ratio",
            args.winner_anchor_output_gradient_ratio,
        ),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{argument_name} must be non-negative and finite.")
    if (
        not math.isfinite(args.auxiliary_output_gradient_scale_max)
        or args.auxiliary_output_gradient_scale_max <= 0
    ):
        raise ValueError(
            "--auxiliary_output_gradient_scale_max must be positive and finite."
        )
    if (
        not math.isfinite(args.dspo_probability_temperature)
        or args.dspo_probability_temperature <= 0
    ):
        raise ValueError("--dspo_probability_temperature must be positive and finite.")
    if not math.isfinite(args.dspo_logit_beta) or args.dspo_logit_beta <= 0:
        raise ValueError("--dspo_logit_beta must be positive and finite.")
    if (
        not math.isfinite(args.dspo_score_correction_scale)
        or not 0 <= args.dspo_score_correction_scale < 1
    ):
        raise ValueError("--dspo_score_correction_scale must be finite and lie in [0, 1).")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
    if args.transition_estimator is None:
        args.transition_estimator = (
            "expected"
            if args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES
            else "pointwise"
        )
    if args.sft and args.preference_loss in RATIO_PREFERENCE_LOSSES:
        raise ValueError(
            f"--preference_loss={args.preference_loss} is only available for preference training, not --sft."
        )
    if args.preference_loss in RATIO_PREFERENCE_LOSSES:
        if not math.isfinite(args.ratio_beta) or args.ratio_beta <= 0:
            raise ValueError("--ratio_beta must be positive.")
        if args.ratio_margin_gradient_scale is not None and (
            not math.isfinite(args.ratio_margin_gradient_scale)
            or args.ratio_margin_gradient_scale <= 0
        ):
            raise ValueError(
                "--ratio_margin_gradient_scale must be positive and finite when set."
            )
        if not math.isfinite(args.max_exp_argument) or args.max_exp_argument <= 0:
            raise ValueError("--max_exp_argument must be positive.")
        if (
            not math.isfinite(args.transition_variance_floor)
            or args.transition_variance_floor <= 0
        ):
            raise ValueError("--transition_variance_floor must be positive.")
        if args.input_perturbation != 0:
            raise ValueError(
                "Ratio Diffusion currently requires --input_perturbation 0 so paired forward noise remains shared."
            )
        if args.noise_offset != 0:
            raise ValueError(
                "Ratio Diffusion currently requires --noise_offset 0 so paired forward noise remains shared."
            )
        if args.sync_pair_augmentations:
            raise NotImplementedError(
                "Synchronized image augmentations are not implemented; leave --sync_pair_augmentations off."
            )
    elif (
        args.ratio_margin_gradient_scale is not None
        or args.reference_anchor_output_gradient_ratio > 0
        or args.winner_anchor_output_gradient_ratio > 0
    ):
        raise ValueError(
            "Ratio margin scaling and auxiliary anchors require a Ratio Diffusion preference loss."
        )
    if args.preference_loss in BREGMAN_PREFERENCE_LOSSES:
        if not math.isfinite(args.bregman_scale) or args.bregman_scale <= 0:
            raise ValueError("--bregman_scale must be positive.")
        if not math.isfinite(args.bregman_lambda):
            raise ValueError("--bregman_lambda must be finite.")
        if abs(args.bregman_lambda + 1.0) < 1e-6:
            raise ValueError("--bregman_lambda values near -1 are not implemented.")
    elif args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES:
        if (
            not math.isfinite(args.mode_seeking_kappa)
            or not 0.0 < args.mode_seeking_kappa < 1.0
        ):
            raise ValueError("--mode_seeking_kappa must lie strictly between 0 and 1.")
    if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES:
        if args.confidence_hidden_dim < 1:
            raise ValueError("--confidence_hidden_dim must be positive.")
        if (
            not math.isfinite(args.confidence_learning_rate)
            or args.confidence_learning_rate <= 0
        ):
            raise ValueError("--confidence_learning_rate must be positive and finite.")
        if (
            args.confidence_timestep_embedding_dim < 2
            or args.confidence_timestep_embedding_dim % 2 != 0
        ):
            raise ValueError(
                "--confidence_timestep_embedding_dim must be an even integer >= 2."
            )
        if (
            not math.isfinite(args.confidence_loss_weight)
            or args.confidence_loss_weight < 0
        ):
            raise ValueError("--confidence_loss_weight must be non-negative and finite.")
        if (
            not math.isfinite(args.confidence_min_policy_weight)
            or not 0.0 < args.confidence_min_policy_weight < 1.0
        ):
            raise ValueError(
                "--confidence_min_policy_weight must lie strictly between 0 and 1."
            )
        if (
            not math.isfinite(args.confidence_label_smoothing)
            or not 0.0 <= args.confidence_label_smoothing < 1.0
        ):
            raise ValueError("--confidence_label_smoothing must lie in [0, 1).")
        if not math.isfinite(args.lrm_score_temperature) or args.lrm_score_temperature <= 0:
            raise ValueError("--lrm_score_temperature must be positive and finite.")
        if args.confidence_policy_signal == "lrm_teacher" and args.confidence_target_source != "precomputed_lrm":
            raise ValueError(
                "--confidence_policy_signal=lrm_teacher requires "
                "--confidence_target_source=precomputed_lrm."
            )
        if args.confidence_target_source == "precomputed_lrm" and args.choice_model:
            raise ValueError(
                "Precomputed LRM scores are tied to the human-labelled jpg_0/jpg_1 order "
                "and cannot be combined with --choice_model relabelling."
            )
        if (
            args.confidence_target_source == "precomputed_lrm"
            and args.pair_label_policy == "legacy"
        ):
            raise ValueError(
                "Precomputed LRM scores require binary human ordering; use "
                "--pair_label_policy=filter or --pair_label_policy=error."
            )
    elif args.confidence_policy_weight_normalization != "none":
        raise ValueError(
            "--confidence_policy_weight_normalization is only valid for a step-aware TBPO loss."
        )
    elif args.confidence_target_source != "reference_mse" or args.confidence_policy_signal != "head":
        raise ValueError(
            "LRM confidence options require a step-aware TBPO preference loss."
        )

    if args.winner_anchor_type == "dspo_score" and args.winner_anchor_output_gradient_ratio <= 0:
        raise ValueError(
            "--winner_anchor_type=dspo_score requires a positive "
            "--winner_anchor_output_gradient_ratio."
        )

    args.ratio_loss_scale = 1.0
    if args.ratio_margin_gradient_scale is not None:
        loss_slope_at_zero = (
            2.0 / args.bregman_scale
            if args.preference_loss in BREGMAN_PREFERENCE_LOSSES
            else 1.0
        )
        args.ratio_loss_scale = preference_loss_margin_scale(
            ratio_beta=args.ratio_beta,
            loss_log_ratio_slope_at_zero=loss_slope_at_zero,
            target_margin_gradient=args.ratio_margin_gradient_scale,
        )
            
    args.train_method = 'sft' if args.sft else 'dpo'
    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


def main():
    
    args = parse_args()
    if os.environ.get("RATIO_OFFLINE_STRICT", "0") == "1":
        model_path = Path(args.pretrained_model_name_or_path).expanduser()
        if not model_path.is_dir():
            raise FileNotFoundError(
                "Offline strict mode requires --pretrained_model_name_or_path to be a local directory: "
                f"{model_path}"
            )
        if args.pretrained_vae_model_name_or_path:
            vae_path_check = Path(args.pretrained_vae_model_name_or_path).expanduser()
            if not vae_path_check.is_dir():
                raise FileNotFoundError(
                    "Offline strict mode requires the VAE to be a local directory: "
                    f"{vae_path_check}"
                )
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                                    subfolder="scheduler")
    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
    
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return 
    if 'turbo' in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)
    
    # SDXL has two text encoders
    if args.sdxl:
        # Load the tokenizers
        if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer_one = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name, subfolder="tokenizer", revision=args.revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    
    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    # 
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        if args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
               tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
            )
            if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_two]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision
        )
        # clone of model
        ref_unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
            subfolder="unet", revision=args.revision
        )
    if args.unet_init:
        print("Initializing unet from", args.unet_init)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_init if args.unet_init else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method == 'dpo': ref_unet.requires_grad_(False)

    ratio_config = None
    confidence_head = None
    if args.preference_loss in RATIO_PREFERENCE_LOSSES:
        if noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError(
                f"{args.preference_loss} requires an epsilon-prediction DDPMScheduler; "
                f"got prediction_type={noise_scheduler.config.prediction_type!r}."
            )
        variance_type = getattr(noise_scheduler.config, "variance_type", "fixed_small")
        if variance_type not in {"fixed_small", "fixed_small_log"}:
            raise ValueError(
                f"{args.preference_loss} requires fixed DDPM posterior variance "
                f"(fixed_small or fixed_small_log), got variance_type={variance_type!r}."
            )
        ratio_alphas_cumprod = noise_scheduler.alphas_cumprod.float()
        invalid_ratio_alphas = (~torch.isfinite(ratio_alphas_cumprod[1:])) | (
            ratio_alphas_cumprod[1:] <= 0
        )
        if torch.any(invalid_ratio_alphas):
            raise ValueError(
                f"{args.preference_loss} requires strictly positive alphas_cumprod for every sampled timestep "
                "t >= 1. Zero-terminal-SNR schedules (for example some Turbo schedulers) are unsupported "
                "by the epsilon-to-clean transition estimator."
            )
        if getattr(unet.config, "out_channels", None) != getattr(unet.config, "in_channels", None):
            raise ValueError(
                f"{args.preference_loss} requires an epsilon-only UNet output, not a learned-variance output head."
            )
        if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES:
            ref_unet.eval()
            confidence_feature_channels = int(unet.config.in_channels) + int(
                unet.config.out_channels
            )
            confidence_head = StepConfidenceHead(
                feature_channels=confidence_feature_channels,
                timestep_embedding_dim=args.confidence_timestep_embedding_dim,
                hidden_dim=args.confidence_hidden_dim,
            )
        ratio_config = {
            "format_version": 5,
            "preference_loss": args.preference_loss,
            "ratio_beta": args.ratio_beta,
            "ratio_margin_gradient_scale": args.ratio_margin_gradient_scale,
            "ratio_loss_scale": args.ratio_loss_scale,
            "theory_version": (
                (
                    (
                        "lrm_calibrated_sa_tbpo_v1"
                        if args.confidence_target_source == "precomputed_lrm"
                        else (
                            "step_aware_power_bregman_tbpo_v1"
                            if args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES
                            else (
                                "dspo_score_anchored_sa_tbpo_v1"
                                if args.winner_anchor_type == "dspo_score"
                                else "sa_tbpo_mvp_v1"
                            )
                        )
                    )
                    if (
                        args.confidence_policy_weight_normalization != "none"
                        or args.ratio_margin_gradient_scale is not None
                        or args.reference_anchor_output_gradient_ratio > 0
                        or args.winner_anchor_output_gradient_ratio > 0
                        or args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES
                        or args.confidence_target_source == "precomputed_lrm"
                    )
                    else "sa_tbpo_v1"
                )
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else (
                    "mode_seeking_ratiodiff_v0.5"
                    if args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES
                    else None
                )
            ),
            "ratio_generator": (
                "scaled_basu"
                if args.preference_loss in BREGMAN_PREFERENCE_LOSSES
                else "power_bregman"
            ),
            "mode_seeking_kappa": (
                args.mode_seeking_kappa
                if args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES
                else None
            ),
            "mode_seeking_loss_convention": (
                "uncentered_eq_27"
                if args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES
                else None
            ),
            "bregman_lambda": (
                args.bregman_lambda
                if args.preference_loss in BREGMAN_PREFERENCE_LOSSES
                else None
            ),
            "bregman_scale": (
                args.bregman_scale
                if args.preference_loss in BREGMAN_PREFERENCE_LOSSES
                else None
            ),
            "ratio_reduction": args.ratio_reduction,
            "transition_estimator": args.transition_estimator,
            "state_correction": args.state_correction,
            "detach_state_correction": args.detach_state_correction,
            "reference_anchor_output_gradient_ratio": (
                args.reference_anchor_output_gradient_ratio
            ),
            "winner_anchor_output_gradient_ratio": (
                args.winner_anchor_output_gradient_ratio
            ),
            "winner_anchor_type": args.winner_anchor_type,
            "dspo_probability_temperature": args.dspo_probability_temperature,
            "dspo_probability_source": args.dspo_probability_source,
            "dspo_logit_beta": args.dspo_logit_beta,
            "dspo_score_correction_scale": args.dspo_score_correction_scale,
            "auxiliary_output_gradient_scale_max": (
                args.auxiliary_output_gradient_scale_max
            ),
            "reference_anchor_definition": "mean_pairs_kl_ref_to_theta",
            "winner_anchor_definition": (
                "dspo_shaped_preferred_score_residual"
                if args.winner_anchor_type == "dspo_score"
                else "preferred_epsilon_denoising_mse"
            ),
            "winner_anchor_probability_detached": (
                True if args.winner_anchor_type == "dspo_score" else None
            ),
            "auxiliary_gradient_budget_space": (
                "global_cross_rank_unet_output_l2_per_microbatch"
            ),
            "auxiliary_gradient_conflict_diagnostic": (
                "global_cross_rank_unet_output_cosine_per_microbatch"
            ),
            "max_exp_argument": args.max_exp_argument,
            "transition_variance_floor": args.transition_variance_floor,
            "input_perturbation": args.input_perturbation,
            "noise_offset": args.noise_offset,
            "sync_pair_augmentations": args.sync_pair_augmentations,
            "confidence_feature_channels": (
                confidence_feature_channels
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_feature_source": (
                "noisy_latent_and_frozen_reference_epsilon"
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_hidden_dim": (
                args.confidence_hidden_dim
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_learning_rate": (
                args.confidence_learning_rate
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_timestep_embedding_dim": (
                args.confidence_timestep_embedding_dim
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_loss_weight": (
                args.confidence_loss_weight
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_min_policy_weight": (
                args.confidence_min_policy_weight
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_policy_weight_normalization": (
                args.confidence_policy_weight_normalization
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_label_smoothing": (
                args.confidence_label_smoothing
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "confidence_target_source": (
                args.confidence_target_source
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            # Keep the v4 descriptive key so existing Q2 checkpoints remain
            # resumable under the unchanged default objective.
            "confidence_target": (
                "reference_pair_denoising_consistency"
                if (
                    args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                    and args.confidence_target_source == "reference_mse"
                )
                else (
                    "precomputed_lrm_soft_preference_probability"
                    if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                    else None
                )
            ),
            "confidence_policy_signal": (
                args.confidence_policy_signal
                if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                else None
            ),
            "lrm_score_0_column": (
                args.lrm_score_0_column
                if args.confidence_target_source == "precomputed_lrm"
                else None
            ),
            "lrm_score_1_column": (
                args.lrm_score_1_column
                if args.confidence_target_source == "precomputed_lrm"
                else None
            ),
            "lrm_score_temperature": (
                args.lrm_score_temperature
                if args.confidence_target_source == "precomputed_lrm"
                else None
            ),
            "lrm_score_input_order": (
                "original_jpg_0_jpg_1_then_human_label_reorder"
                if args.confidence_target_source == "precomputed_lrm"
                else None
            ),
            "lrm_timestep_bin_mapping": (
                "floor(t_times_K_over_T)"
                if args.confidence_target_source == "precomputed_lrm"
                else None
            ),
            "detach_policy_weight": (
                True if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES else None
            ),
            "pair_label_policy": args.pair_label_policy,
            "pair_label_source": (
                f"choice_model:{args.choice_model}"
                if args.choice_model
                else "human_label_0"
            ),
            "no_hflip": args.no_hflip,
            "random_crop": args.random_crop,
            "proportion_empty_prompts": args.proportion_empty_prompts,
        }

    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        logger.warning(
            "xformers is not installed; using PyTorch native scaled-dot-product "
            "attention. This is the supported offline/B200 fallback."
        )

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    # 
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):

            for model in models:
                model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    if isinstance(model, StepConfidenceHead):
                        torch.save(
                            model.state_dict(),
                            os.path.join(output_dir, "step_confidence_head.pt"),
                        )
                    else:
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
            if accelerator.is_main_process and ratio_config is not None:
                with open(os.path.join(output_dir, "ratio_diffusion_config.json"), "w", encoding="utf-8") as config_file:
                    json.dump(ratio_config, config_file, indent=2, sort_keys=True)

        def load_model_hook(models, input_dir):

            saved_ratio_config_path = os.path.join(input_dir, "ratio_diffusion_config.json")
            if os.path.isfile(saved_ratio_config_path):
                with open(saved_ratio_config_path, "r", encoding="utf-8") as config_file:
                    saved_ratio_config = json.load(config_file)
                if ratio_config is None:
                    raise ValueError(
                        "Refusing to resume a Ratio Diffusion checkpoint with a non-ratio objective."
                    )
                resume_fields = (
                    "preference_loss",
                    "ratio_beta",
                    "theory_version",
                    "mode_seeking_kappa",
                    "mode_seeking_loss_convention",
                    "bregman_lambda",
                    "bregman_scale",
                    "ratio_reduction",
                    "transition_estimator",
                    "state_correction",
                    "detach_state_correction",
                    "max_exp_argument",
                    "transition_variance_floor",
                    "input_perturbation",
                    "noise_offset",
                    "sync_pair_augmentations",
                    "confidence_feature_channels",
                    "confidence_feature_source",
                    "confidence_hidden_dim",
                    "confidence_learning_rate",
                    "confidence_timestep_embedding_dim",
                    "confidence_loss_weight",
                    "confidence_min_policy_weight",
                    "confidence_label_smoothing",
                    "confidence_target",
                    "detach_policy_weight",
                )
                mismatches = {
                    field: (saved_ratio_config.get(field), ratio_config.get(field))
                    for field in resume_fields
                    if saved_ratio_config.get(field) != ratio_config.get(field)
                }
                # Versions 2/3 predate these objective controls. Their absence
                # therefore means the exact legacy defaults, not "unknown".
                v5_objective_defaults = {
                    "ratio_margin_gradient_scale": None,
                    "ratio_loss_scale": 1.0,
                    "ratio_generator": (
                        "scaled_basu"
                        if args.preference_loss in BREGMAN_PREFERENCE_LOSSES
                        else "power_bregman"
                    ),
                    "reference_anchor_output_gradient_ratio": 0.0,
                    "winner_anchor_output_gradient_ratio": 0.0,
                    "winner_anchor_type": "mse",
                    "dspo_probability_temperature": 1.0,
                    "dspo_probability_source": "mse_preference",
                    "dspo_logit_beta": 0.01,
                    "dspo_score_correction_scale": 0.25,
                    "auxiliary_output_gradient_scale_max": 1000.0,
                    "confidence_policy_weight_normalization": (
                        "none"
                        if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                        else None
                    ),
                    "confidence_target_source": (
                        "reference_mse"
                        if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                        else None
                    ),
                    "confidence_policy_signal": (
                        "head"
                        if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                        else None
                    ),
                    "lrm_score_0_column": None,
                    "lrm_score_1_column": None,
                    "lrm_score_temperature": None,
                }
                for field, legacy_default in v5_objective_defaults.items():
                    saved_value = saved_ratio_config.get(field, legacy_default)
                    current_value = ratio_config.get(field)
                    if saved_value != current_value:
                        mismatches[field] = (saved_value, current_value)

                # Data/protocol identity was not recorded by older checkpoints,
                # so enforce it once both sides use schema v4.
                if int(saved_ratio_config.get("format_version", 1)) >= 4:
                    for field in (
                        "pair_label_policy",
                        "pair_label_source",
                        "no_hflip",
                        "random_crop",
                        "proportion_empty_prompts",
                    ):
                        saved_value = saved_ratio_config.get(field)
                        current_value = ratio_config.get(field)
                        if saved_value != current_value:
                            mismatches[field] = (saved_value, current_value)
                if mismatches:
                    details = ", ".join(
                        f"{field}: saved={saved!r}, current={current!r}"
                        for field, (saved, current) in mismatches.items()
                    )
                    raise ValueError(
                        "Refusing to resume with a different Ratio Diffusion configuration: "
                        + details
                    )
            elif ratio_config is not None:
                raise FileNotFoundError(
                    "Ratio Diffusion resume requires ratio_diffusion_config.json in the checkpoint."
                )

            while models:
                # pop models so that they are not loaded again
                model = models.pop()
                model = accelerator.unwrap_model(model)

                if isinstance(model, StepConfidenceHead):
                    confidence_path = os.path.join(
                        input_dir, "step_confidence_head.pt"
                    )
                    if not os.path.isfile(confidence_path):
                        raise FileNotFoundError(
                            "Step-aware TBPO checkpoint is missing step_confidence_head.pt."
                        )
                    confidence_state = torch.load(
                        confidence_path, map_location="cpu"
                    )
                    model.load_state_dict(confidence_state)
                    continue

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL")
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    unet_parameters = list(unet.parameters())
    trainable_parameters = list(unet_parameters)
    optimizer_parameter_groups = [
        {"params": unet_parameters, "lr": args.learning_rate}
    ]
    if confidence_head is not None:
        confidence_parameters = list(confidence_head.parameters())
        trainable_parameters.extend(confidence_parameters)
        optimizer_parameter_groups.append(
            {
                "params": confidence_parameters,
                "lr": args.confidence_learning_rate,
            }
        )

    if args.use_adafactor or args.sdxl:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(optimizer_parameter_groups,
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        
        
        
    streaming_mode = bool(args.pickapic_streaming_manifest)
    if streaming_mode:
        if not args.pickapic_stream_cache:
            raise ValueError("--pickapic_stream_cache is required with streaming mode")
        if args.dataloader_num_workers != 0:
            raise ValueError("streaming mode requires --dataloader_num_workers=0")
        if args.pair_label_policy == "filter" and not args.choice_model:
            raise ValueError(
                "Streaming mode cannot safely skip a rank-dependent number of tie labels. "
                "Prefilter the manifest to binary labels, then use --pair_label_policy=error."
            )
        dataset = None
        column_names = list(DATASET_NAME_MAPPING["yuvalkirstain/pickapic_v2"])
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files[args.split] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if not streaming_mode:
        column_names = dataset[args.split].column_names
    if args.confidence_target_source == "precomputed_lrm":
        if streaming_mode:
            raise ValueError(
                "Precomputed LRM score columns are currently supported only by the "
                "map-style parquet/Hugging Face dataset path, not streaming mode."
            )
        missing_lrm_columns = [
            name
            for name in (args.lrm_score_0_column, args.lrm_score_1_column)
            if name not in column_names
        ]
        if missing_lrm_columns:
            raise ValueError(
                "Missing precomputed LRM score column(s): "
                + ", ".join(missing_lrm_columns)
                + ". Each row must provide a scalar, K timestep-bin scores, or T per-timestep scores for each image."
            )

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if (
        (args.dataset_name is not None and 'pickapic' in args.dataset_name)
        or (args.train_method == 'dpo')
    ):
        pass
    elif args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    
    ##### START BIG OLD DATASET BLOCK #####
    
    #### START PREPROCESSING/COLLATION ####
    if args.train_method == 'dpo':
        print("Ignoring image_column variable, reading from jpg_0 and jpg_1")
        def preprocess_train(examples):
            all_pixel_values = []
            for col_name in ['jpg_0', 'jpg_1']:
                images = [Image.open(io.BytesIO(im_bytes)).convert("RGB")
                            for im_bytes in examples[col_name]]
                pixel_values = [train_transforms(image) for image in images]
                all_pixel_values.append(pixel_values)
            # Double on channel dim, jpg_y then jpg_w
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []
            preferred_lrm_scores = []
            rejected_lrm_scores = []
            if args.confidence_target_source == "precomputed_lrm":
                lrm_score_iterator = zip(
                    examples[args.lrm_score_0_column],
                    examples[args.lrm_score_1_column],
                )
            else:
                lrm_score_iterator = [None] * len(examples['label_0'])
            for im_tup, label_0, lrm_scores in zip(
                im_tup_iterator, examples['label_0'], lrm_score_iterator
            ):
                if (
                    not args.choice_model
                    and args.pair_label_policy != "legacy"
                    and label_0 not in (0, 1)
                ):
                    raise ValueError(
                        "Encountered a non-binary label_0 after applying "
                        f"--pair_label_policy={args.pair_label_policy}."
                    )
                if label_0==0 and (not args.choice_model): # don't want to flip things if using choice_model for AI feedback
                    im_tup = im_tup[::-1]
                    if lrm_scores is not None:
                        lrm_scores = lrm_scores[::-1]
                combined_im = torch.cat(im_tup, dim=0) # no batch dim
                combined_pixel_values.append(combined_im)
                if lrm_scores is not None:
                    preferred_lrm_scores.append(lrm_scores[0])
                    rejected_lrm_scores.append(lrm_scores[1])
            examples["pixel_values"] = combined_pixel_values
            if args.confidence_target_source == "precomputed_lrm":
                examples["lrm_preferred_scores"] = preferred_lrm_scores
                examples["lrm_rejected_scores"] = rejected_lrm_scores
            # SDXL takes raw prompts
            if not args.sdxl: examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            if args.confidence_target_source == "precomputed_lrm":
                preferred_scores = [
                    torch.as_tensor(example["lrm_preferred_scores"], dtype=torch.float32).reshape(-1)
                    for example in examples
                ]
                rejected_scores = [
                    torch.as_tensor(example["lrm_rejected_scores"], dtype=torch.float32).reshape(-1)
                    for example in examples
                ]
                try:
                    return_d["lrm_preferred_scores"] = torch.stack(preferred_scores)
                    return_d["lrm_rejected_scores"] = torch.stack(rejected_scores)
                except RuntimeError as error:
                    raise ValueError(
                        "All precomputed LRM score rows must have the same number of timestep bins."
                    ) from error
                
            if args.choice_model:
                # If using AIF then deliver image data for choice model to determine if should flip pixel values
                for k in ['jpg_0', 'jpg_1']:
                    return_d[k] = [Image.open(io.BytesIO( example[k])).convert("RGB")
                                   for example in examples]
                return_d["caption"] = [example["caption"] for example in examples] 
            return return_d
         
        if args.choice_model:
            # TODO: Fancy way of doing this?
            if args.choice_model == 'hps':
                from utils.hps_utils import Selector
            elif args.choice_model == 'clip':
                from utils.clip_utils import Selector
            elif args.choice_model == 'pickscore':
                from utils.pickscore_utils import Selector
            elif args.choice_model == 'aes':
                from utils.aes_utils import Selector
            selector = Selector('cpu' if args.sdxl else accelerator.device)

            def do_flip(jpg0, jpg1, prompt):
                scores = selector.score([jpg0, jpg1], prompt)
                return scores[1] > scores[0]
            def choice_model_says_flip(batch):
                assert len(batch['caption'])==1 # Can switch to iteration but not needed for nwo
                return do_flip(batch['jpg_0'][0], batch['jpg_1'][0], batch['caption'][0])
    elif args.train_method == 'sft':
        def preprocess_train(examples):
            if args.dataset_name is not None and 'pickapic' in args.dataset_name:
                images = []
                # Probably cleaner way to do this iteration
                for im_0_bytes, im_1_bytes, label_0 in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0']):
                    assert label_0 in (0, 1)
                    im_bytes = im_0_bytes if label_0==1 else im_1_bytes
                    images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
            else:
                images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            if not args.sdxl: examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d
    #### END PREPROCESSING/COLLATION ####
    
    ### DATASET #####
    if streaming_mode:
        from hessian.streaming_pickapic import PickAPicStreamingDataset

        def configured_resume_step() -> int:
            if not args.resume_from_checkpoint:
                return 0
            if args.resume_from_checkpoint == "latest":
                if not os.path.isdir(args.output_dir):
                    return 0
                candidates = [
                    name for name in os.listdir(args.output_dir)
                    if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()
                ]
                return max((int(name.split("-")[-1]) for name in candidates), default=0)
            name = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
            path = (
                args.resume_from_checkpoint
                if os.path.isdir(args.resume_from_checkpoint)
                else os.path.join(args.output_dir, name)
            )
            return int(name.split("-")[-1]) if os.path.isdir(path) else 0

        resume_optimizer_step = configured_resume_step()
        resume_samples_per_rank = (
            resume_optimizer_step * args.gradient_accumulation_steps * args.train_batch_size
        )
        raw_stream_dataset = PickAPicStreamingDataset(
            manifest_path=args.pickapic_streaming_manifest,
            local_data_dir=args.train_data_dir,
            stream_cache_root=args.pickapic_stream_cache,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            seed=args.seed,
            start_sample=resume_samples_per_rank,
            retries=args.streaming_retries,
            pad_to_multiple=args.gradient_accumulation_steps * args.train_batch_size,
            prefetch_next_shard=args.streaming_prefetch_next_shard,
        )

        class TransformedStreamingDataset(torch.utils.data.IterableDataset):
            def __len__(self):
                return len(raw_stream_dataset)

            def __iter__(self):
                for raw in raw_stream_dataset:
                    transformed = preprocess_train({key: [value] for key, value in raw.items()})
                    sample = dict(raw)
                    sample["pixel_values"] = transformed["pixel_values"][0]
                    if not args.sdxl:
                        sample["input_ids"] = transformed["input_ids"][0]
                    yield sample

        train_dataset = TransformedStreamingDataset()
    else:
        with accelerator.main_process_first():
            invalid_label_count = 0
            dataset_name_is_pickapic = (
                args.dataset_name is not None and 'pickapic' in args.dataset_name
            )
            filter_pair_labels = dataset_name_is_pickapic or (
                args.pair_label_policy == "filter" and not args.choice_model
            )
            validate_pair_labels = (
                args.pair_label_policy == "error" and not args.choice_model
            )
            if filter_pair_labels or validate_pair_labels:
                if 'label_0' not in dataset[args.split].column_names:
                    raise ValueError(
                        "Binary pair-label validation requires a label_0 dataset column."
                    )
                orig_len = dataset[args.split].num_rows
                not_split_idx = [i for i,label_0 in enumerate(dataset[args.split]['label_0'])
                                 if label_0 in (0,1) ]
                invalid_label_count = orig_len - len(not_split_idx)
                if validate_pair_labels and invalid_label_count:
                    raise ValueError(
                        f"Found {invalid_label_count}/{orig_len} non-binary label_0 values; "
                        "use --pair_label_policy=filter or prefilter the dataset."
                    )
                if filter_pair_labels:
                    dataset[args.split] = dataset[args.split].select(not_split_idx)
                    print(
                        f"Eliminated {invalid_label_count}/{orig_len} non-binary pair labels"
                    )

            if dataset_name_is_pickapic:
                # Below if if want to train on just the Dreamlike vs dreamlike pairs
                if args.dreamlike_pairs_only:
                    orig_len = dataset[args.split].num_rows
                    dream_like_idx = [i for i,(m0,m1) in enumerate(zip(dataset[args.split]['model_0'],
                                                                       dataset[args.split]['model_1']))
                                      if ( ('dream' in m0) and ('dream' in m1) )]
                    dataset[args.split] = dataset[args.split].select(dream_like_idx)
                    new_len = dataset[args.split].num_rows
                    print(f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic")

            if args.max_train_samples is not None:
                available_samples = dataset[args.split].num_rows
                selected_samples = min(args.max_train_samples, available_samples)
                if selected_samples < args.max_train_samples:
                    print(
                        f"Requested {args.max_train_samples} samples but only "
                        f"{available_samples} remain after filtering; using all available pairs."
                    )
                dataset[args.split] = dataset[args.split].shuffle(seed=args.seed).select(
                    range(selected_samples)
                )
            if ratio_config is not None:
                ratio_config.update(
                    {
                        "nonbinary_pair_labels_removed": (
                            invalid_label_count if filter_pair_labels else 0
                        ),
                        "train_samples_after_filter_and_truncation": (
                            dataset[args.split].num_rows
                        ),
                        "max_train_samples_requested": args.max_train_samples,
                    }
                )
            # Set the training transforms
            train_dataset = dataset[args.split].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split=='train' and not streaming_mode),
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0 if streaming_mode else args.dataloader_num_workers,
        drop_last=True
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    if streaming_mode:
        # The iterable is already partitioned by rank. Preparing this loader
        # would shard it a second time, so only prepare model/optimizer/scheduler.
        if confidence_head is None:
            unet, optimizer, lr_scheduler = accelerator.prepare(
                unet, optimizer, lr_scheduler
            )
        else:
            unet, confidence_head, optimizer, lr_scheduler = accelerator.prepare(
                unet, confidence_head, optimizer, lr_scheduler
            )
    else:
        if confidence_head is None:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        else:
            (
                unet,
                confidence_head,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                unet,
                confidence_head,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

        
    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        print("offload vae (this actually stays as CPU)")
        vae = accelerate.cpu_offload(vae)
        print("Offloading text encoders to cpu")
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            print("offload ref_unet")
            ref_unet = accelerate.cpu_offload(ref_unet)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_step = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            if streaming_mode:
                # The rank-partitioned iterable starts at the checkpoint's
                # exact per-rank sample offset, so no dataloader replay is needed.
                first_epoch = 0
                resume_step = 0
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Keep ratio diagnostics for the whole gradient-accumulation window.  The
    # previous implementation only reported its final microbatch, which could
    # hide clipping or non-finite spikes from earlier microbatches.
    ratio_diagnostic_window = []

    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if confidence_head is not None:
            confidence_head.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        model_mse_accumulated = 0.0
        reference_mse_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            if streaming_mode:
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True)
                if "input_ids" in batch:
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
            accumulation_window_start = (
                step // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            accumulation_window_size = min(
                args.gradient_accumulation_steps,
                len(train_dataloader) - accumulation_window_start,
            )
            # accelerate==0.20.2 (the repository pin) accepts exactly one
            # model here. The confidence head shares the same accelerated
            # optimizer, so its gradients still accumulate correctly; its DDP
            # reducer may synchronize each microbatch on this legacy version.
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method == 'dpo':
                    # y_w and y_l were concatenated along channel dimension
                    feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                    # If using AIF then we haven't ranked yet so do so now
                    # Only implemented for BS=1 (assert-protected)
                    if args.choice_model:
                        if choice_model_says_flip(batch):
                            feed_pixel_values = feed_pixel_values.flip(0)
                elif args.train_method == 'sft':
                    feed_pixel_values = batch["pixel_values"]
                
                #### Diffusion Stuff ####
                # encode pixels --> latents
                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                if args.preference_loss in RATIO_PREFERENCE_LOSSES:
                    # Keep the original preferred-first/rejected-second packing.  Common
                    # random numbers are mandatory here: one t and one forward epsilon per pair.
                    if latents.shape[0] % 2 != 0:
                        raise ValueError(
                            f"{args.preference_loss} requires an even latent batch packed as preferred then rejected pairs."
                        )
                    pair_batch_size = latents.shape[0] // 2
                    num_train_timesteps = noise_scheduler.config.num_train_timesteps
                    if num_train_timesteps < 2:
                        raise ValueError(
                            f"{args.preference_loss} requires a scheduler with at least two training timesteps."
                        )
                    pair_timesteps = torch.randint(
                        low=1,
                        high=num_train_timesteps,
                        size=(pair_batch_size,),
                        device=latents.device,
                        dtype=torch.long,
                    )
                    timesteps = pair_timesteps.repeat(2)
                    pair_forward_noise = torch.randn_like(latents[:pair_batch_size])
                    noise = pair_forward_noise.repeat((2,) + (1,) * (latents.ndim - 1))
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                else:
                    # Original Diffusion-DPO/SFT noising path.  Keep this branch unchanged
                    # so --preference_loss dpo remains the reproducible baseline.
                    noise = torch.randn_like(latents)
                    # variants of noising
                    if args.noise_offset: # haven't tried yet
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if args.input_perturbation: # haven't tried yet
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                        
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    # only first 20% timesteps for SDXL refiner
                    if 'refiner' in args.pretrained_model_name_or_path:
                        timesteps = timesteps % 200
                    elif 'turbo' in args.pretrained_model_name_or_path:
                        timesteps_0_to_3 = timesteps % 4
                        timesteps = 250 * timesteps_0_to_3 + 249
                    
                    if args.train_method == 'dpo': # make timesteps and noise same for pairs in DPO
                        timesteps = timesteps.chunk(2)[0].repeat(2)
                        noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents,
                                                              new_noise if args.input_perturbation else noise,
                                                              timesteps)
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        # Need to compute "time_ids" https://github.com/huggingface/diffusers/blob/v0.20.0-release/examples/text_to_image/train_text_to_image_sdxl.py#L969
                        # for SDXL-base these are torch.tensor([args.resolution, args.resolution, *crop_coords_top_left, *target_size))
                        if 'refiner' in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          6.0], # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        else: # SDXL-base
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          args.resolution, 
                                                         args.resolution],
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(batch, 
                                                          text_encoders,
                                                           tokenizers,
                                                           args.proportion_empty_prompts, 
                                                          caption_column='caption',
                                                           is_train=True,
                                                          )
                    if args.train_method == 'dpo':
                        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                    unet_added_conditions = {"time_ids": add_time_ids,
                                            "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                else: # sd1.5
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    if args.train_method == 'dpo':
                        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####
                        
                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise
               
                # Make the prediction from the model we're learning
                model_batch_args = (noisy_latents,
                                    timesteps, 
                                    prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states)
                added_cond_kwargs = unet_added_conditions if args.sdxl else None
                
                model_pred = unet(
                                *model_batch_args,
                                  added_cond_kwargs = added_cond_kwargs
                                 ).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == 'sft': # SFT, casting for F.mse_loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.train_method == 'dpo':
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        ref_pred = ref_unet(
                                    *model_batch_args,
                                      added_cond_kwargs = added_cond_kwargs
                                     ).sample.detach()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        raw_ref_loss = ref_losses.mean()    
                        
                    dpo_loss, inside_term = diffusion_dpo_loss_from_per_sample_losses(
                        model_losses, ref_losses, args.beta_dpo
                    )
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    if args.preference_loss == "dpo":
                        loss = dpo_loss
                    elif args.preference_loss in RATIO_PREFERENCE_LOSSES:
                        # The Ratio Diffusion estimator is a reverse-transition density
                        # ratio.  It intentionally does not reuse Diffusion-DPO's MSE logit.
                        coefficients = ddpm_posterior_coefficients(
                            alphas_cumprod=noise_scheduler.alphas_cumprod,
                            timesteps=timesteps,
                            sample_shape=noisy_latents.shape,
                            variance_floor=args.transition_variance_floor,
                        )
                        clean_hat_theta = predict_clean_from_epsilon(
                            noisy_latents=noisy_latents,
                            epsilon_prediction=model_pred,
                            alpha_t=coefficients.alpha_t,
                            sigma_t=coefficients.sigma_t,
                        )
                        clean_hat_ref = predict_clean_from_epsilon(
                            noisy_latents=noisy_latents,
                            epsilon_prediction=ref_pred,
                            alpha_t=coefficients.alpha_t,
                            sigma_t=coefficients.sigma_t,
                        )
                        mean_theta = transition_mean(noisy_latents, clean_hat_theta, coefficients)
                        mean_ref = transition_mean(noisy_latents, clean_hat_ref, coefficients)

                        if args.transition_estimator == "pointwise":
                            # A shared eta is another common-random-number control variate
                            # for the winner-minus-loser transition comparison.
                            pair_posterior_noise = torch.randn_like(
                                latents[:pair_batch_size], dtype=torch.float32
                            )
                            posterior_noise = pair_posterior_noise.repeat(
                                (2,) + (1,) * (latents.ndim - 1)
                            )
                            action_s, true_mean = sample_true_posterior_action(
                                noisy_latents=noisy_latents,
                                clean_latents=latents,
                                posterior_noise=posterior_noise,
                                coefficients=coefficients,
                            )
                            ratio_delta = transition_log_uplift(
                                action=action_s,
                                current_mean=mean_theta,
                                reference_mean=mean_ref,
                                variance=coefficients.variance,
                                reduction=args.ratio_reduction,
                            )
                        else:
                            # This is an expected-log-ratio/Rao-Blackwellized ablation,
                            # rather than the nonlinear pointwise Bregman objective.
                            true_mean = true_posterior_mean(noisy_latents, latents, coefficients)
                            ratio_delta = expected_transition_log_uplift(
                                true_mean=true_mean,
                                current_mean=mean_theta,
                                reference_mean=mean_ref,
                                variance=coefficients.variance,
                                reduction=args.ratio_reduction,
                            )

                        ratio_delta_w, ratio_delta_l = ratio_delta.float().chunk(2)
                        needs_attached_reference_kl = (
                            args.state_correction == "exact_gaussian"
                            or args.reference_anchor_output_gradient_ratio > 0
                        )
                        if needs_attached_reference_kl:
                            attached_reference_kl = equal_covariance_gaussian_kl(
                                reference_mean=mean_ref,
                                current_mean=mean_theta,
                                variance=coefficients.variance,
                                reduction=args.ratio_reduction,
                            ).float()
                        else:
                            attached_reference_kl = torch.zeros_like(ratio_delta)

                        if args.state_correction == "exact_gaussian":
                            ratio_kl_w, ratio_kl_l = attached_reference_kl.chunk(2)
                            if args.detach_state_correction:
                                ratio_kl_w = ratio_kl_w.detach()
                                ratio_kl_l = ratio_kl_l.detach()
                        else:
                            ratio_kl_w = torch.zeros_like(ratio_delta_w)
                            ratio_kl_l = torch.zeros_like(ratio_delta_l)

                        ratio_log_omega = ratio_kl_l - ratio_kl_w
                        ratio_raw_margin = (
                            ratio_delta_l - ratio_delta_w + ratio_log_omega
                        )
                        ratio_log_r = args.ratio_beta * ratio_raw_margin
                        any_rank_nonfinite = accelerator.gather(
                            (~torch.isfinite(ratio_log_r)).any().float().reshape(1)
                        ).max()
                        if bool(any_rank_nonfinite):
                            raise FloatingPointError(
                                "Non-finite state-corrected log-ratio detected on at least one rank. "
                                "Lower ratio_beta or inspect the transition/state-correction terms."
                            )
                        if args.preference_loss in BREGMAN_PREFERENCE_LOSSES:
                            ratio_unscaled_per_pair_loss = scaled_basu_loss_from_log_ratio(
                                log_ratio=ratio_log_r,
                                bregman_lambda=args.bregman_lambda,
                                bregman_scale=args.bregman_scale,
                                max_exp_argument=args.max_exp_argument,
                            )
                            ratio_exponent_clipped = scaled_basu_exponent_clip_fraction(
                                log_ratio=ratio_log_r,
                                bregman_lambda=args.bregman_lambda,
                                max_exp_argument=args.max_exp_argument,
                            )
                        else:
                            ratio_unscaled_per_pair_loss = mode_seeking_power_loss_from_log_ratio(
                                log_ratio=ratio_log_r,
                                kappa=args.mode_seeking_kappa,
                                max_exp_argument=args.max_exp_argument,
                            )
                            ratio_exponent_clipped = mode_seeking_exponent_clip_fraction(
                                log_ratio=ratio_log_r,
                                kappa=args.mode_seeking_kappa,
                                max_exp_argument=args.max_exp_argument,
                            )
                        ratio_per_pair_loss = (
                            float(args.ratio_loss_scale)
                            * ratio_unscaled_per_pair_loss.float()
                        )
                        lrm_teacher_probability = None
                        lrm_teacher_score_gap = None
                        if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES:
                            if confidence_head is None:
                                raise RuntimeError(
                                    "A step-aware TBPO loss requires an initialized confidence head."
                                )
                            # Use the noisy state together with the frozen reference
                            # UNet response as compact h_t features. This makes the
                            # learned positive weights exogenous to theta and keeps the
                            # confidence task stationary while theta is optimized.
                            confidence_features = torch.cat(
                                [noisy_latents.detach(), ref_pred.detach()], dim=1
                            )
                            step_confidence = confidence_head(
                                features=confidence_features,
                                pair_timesteps=pair_timesteps,
                                num_train_timesteps=num_train_timesteps,
                            )
                            if args.confidence_target_source == "reference_mse":
                                confidence_reference_losses = (
                                    ref_pred.float() - target.float()
                                ).square().mean(dim=[1, 2, 3])
                                confidence_target = step_preference_consistency_target(
                                    reference_losses=confidence_reference_losses.detach(),
                                    label_smoothing=args.confidence_label_smoothing,
                                )
                                policy_confidence = None
                            else:
                                selected_lrm_w, selected_lrm_l = select_timestep_pair_scores(
                                    preferred_scores=batch["lrm_preferred_scores"].to(
                                        device=ratio_log_r.device
                                    ),
                                    rejected_scores=batch["lrm_rejected_scores"].to(
                                        device=ratio_log_r.device
                                    ),
                                    timesteps=pair_timesteps,
                                    num_train_timesteps=num_train_timesteps,
                                )
                                lrm_teacher_probability = latent_teacher_preference_probability(
                                    preferred_scores=selected_lrm_w,
                                    rejected_scores=selected_lrm_l,
                                    temperature=args.lrm_score_temperature,
                                ).detach()
                                lrm_teacher_score_gap = (
                                    selected_lrm_w.float() - selected_lrm_l.float()
                                ).detach()
                                confidence_target = (
                                    lrm_teacher_probability
                                    * (1.0 - args.confidence_label_smoothing)
                                    + 0.5 * args.confidence_label_smoothing
                                )
                                policy_confidence = (
                                    lrm_teacher_probability
                                    if args.confidence_policy_signal == "lrm_teacher"
                                    else None
                                )
                            policy_weight_normalizer = None
                            if (
                                args.confidence_policy_weight_normalization
                                == "global_microbatch_mean"
                            ):
                                policy_signal_for_weight = (
                                    policy_confidence
                                    if policy_confidence is not None
                                    else step_confidence.detach().float()
                                )
                                raw_policy_weights = args.confidence_min_policy_weight + (
                                    1.0 - args.confidence_min_policy_weight
                                ) * policy_signal_for_weight
                                policy_weight_normalizer = accelerator.gather(
                                    raw_policy_weights
                                ).mean()
                            step_aware_output = step_aware_preference_loss(
                                per_pair_loss=ratio_per_pair_loss,
                                confidence=step_confidence,
                                confidence_target=confidence_target,
                                confidence_loss_weight=args.confidence_loss_weight,
                                minimum_policy_weight=args.confidence_min_policy_weight,
                                policy_weight_normalizer=policy_weight_normalizer,
                                policy_confidence=policy_confidence,
                            )
                            ratio_policy_loss = step_aware_output.policy_loss
                            confidence_objective = (
                                float(args.confidence_loss_weight)
                                * step_aware_output.confidence_loss
                            )
                        else:
                            ratio_policy_loss = ratio_per_pair_loss.mean()
                            confidence_objective = ratio_policy_loss.new_zeros(())

                        # Calibrate auxiliary coefficients by output-space gradients.
                        # The norms are combined across ranks so every worker uses the
                        # same detached multiplier. This avoids a second UNet backward.
                        def global_output_gradient_norm(local_gradient):
                            per_rank_squared_norm = accelerator.gather(
                                local_gradient.float().square().sum().reshape(1)
                            )
                            return per_rank_squared_norm.sum().sqrt().detach()

                        def global_output_gradient_cosine(
                            first_gradient,
                            second_gradient,
                            first_norm,
                            second_norm,
                        ):
                            global_dot = accelerator.gather(
                                (first_gradient.float() * second_gradient.float())
                                .sum()
                                .reshape(1)
                            ).sum()
                            denominator = first_norm.float() * second_norm.float()
                            cosine = torch.where(
                                denominator > 0,
                                global_dot / denominator.clamp_min(1e-12),
                                torch.zeros_like(global_dot),
                            )
                            return cosine.clamp(-1.0, 1.0).detach()

                        ratio_primary_output_gradient = loss_output_gradient(
                            ratio_policy_loss, model_pred
                        )
                        ratio_primary_output_gradient_norm = global_output_gradient_norm(
                            ratio_primary_output_gradient
                        )
                        reference_anchor_loss = ratio_policy_loss.new_zeros(())
                        reference_anchor_output_gradient_norm = (
                            ratio_policy_loss.new_zeros(())
                        )
                        reference_anchor_output_gradient_cosine = (
                            ratio_policy_loss.new_zeros(())
                        )
                        reference_anchor_scale = ratio_policy_loss.new_zeros(())
                        if args.reference_anchor_output_gradient_ratio > 0:
                            reference_anchor_loss = symmetric_reference_kl_anchor(
                                attached_reference_kl
                            )
                            reference_anchor_output_gradient = loss_output_gradient(
                                reference_anchor_loss, model_pred
                            )
                            reference_anchor_output_gradient_norm = (
                                global_output_gradient_norm(
                                    reference_anchor_output_gradient
                                )
                            )
                            reference_anchor_output_gradient_cosine = (
                                global_output_gradient_cosine(
                                    ratio_primary_output_gradient,
                                    reference_anchor_output_gradient,
                                    ratio_primary_output_gradient_norm,
                                    reference_anchor_output_gradient_norm,
                                )
                            )
                            reference_anchor_scale = auxiliary_output_gradient_scale(
                                primary_gradient_norm=ratio_primary_output_gradient_norm,
                                auxiliary_gradient_norm=(
                                    reference_anchor_output_gradient_norm
                                ),
                                target_ratio=(
                                    args.reference_anchor_output_gradient_ratio
                                ),
                                maximum_scale=(
                                    args.auxiliary_output_gradient_scale_max
                                ),
                            )

                        winner_anchor_loss = ratio_policy_loss.new_zeros(())
                        winner_anchor_output_gradient_norm = (
                            ratio_policy_loss.new_zeros(())
                        )
                        winner_anchor_output_gradient_cosine = (
                            ratio_policy_loss.new_zeros(())
                        )
                        winner_anchor_scale = ratio_policy_loss.new_zeros(())
                        dspo_preference_logit = None
                        dspo_preferred_probability = None
                        dspo_actionability = None
                        dspo_residual_jacobian_scale = None
                        dspo_correction_to_data_ratio = None
                        if args.winner_anchor_output_gradient_ratio > 0:
                            if args.winner_anchor_type == "dspo_score":
                                if args.dspo_probability_source == "mse_preference":
                                    # DSPO's SD1.5 logit is a difference of
                                    # winner/loser denoising MSE differences.
                                    # Recompute it in fp32: the recommended
                                    # beta_D values are small enough that bf16
                                    # cancellation would otherwise erase much
                                    # of the gate's variation.
                                    dspo_model_losses = (
                                        model_pred.detach().float() - target.detach().float()
                                    ).square().mean(dim=[1, 2, 3])
                                    dspo_reference_losses = (
                                        ref_pred.float() - target.float()
                                    ).square().mean(dim=[1, 2, 3])
                                    dspo_model_w, dspo_model_l = dspo_model_losses.chunk(2)
                                    dspo_reference_w, dspo_reference_l = (
                                        dspo_reference_losses.chunk(2)
                                    )
                                    model_mse_difference = dspo_model_w - dspo_model_l
                                    reference_mse_difference = (
                                        dspo_reference_w - dspo_reference_l
                                    )
                                    dspo_preference_logit = (
                                        -0.5
                                        * args.dspo_logit_beta
                                        * (model_mse_difference - reference_mse_difference)
                                    )
                                else:
                                    # ratio_log_r is rejected-over-preferred, hence the
                                    # sign reversal for preferred probability.
                                    dspo_preference_logit = -ratio_log_r.detach().float()
                                dspo_preferred_probability = torch.sigmoid(
                                    dspo_preference_logit / args.dspo_probability_temperature
                                )
                                dspo_actionability = 1.0 - dspo_preferred_probability
                                dspo_residual_jacobian_scale = (
                                    1.0
                                    - args.dspo_score_correction_scale
                                    * dspo_actionability
                                )
                                dspo_broadcast_shape = (
                                    dspo_actionability.shape[0],
                                ) + (1,) * (model_pred.ndim - 1)
                                dspo_data_residual = (
                                    model_pred[: dspo_actionability.shape[0]].detach().float()
                                    - target[: dspo_actionability.shape[0]].detach().float()
                                )
                                dspo_score_correction = (
                                    args.dspo_score_correction_scale
                                    * dspo_actionability.reshape(dspo_broadcast_shape)
                                    * (
                                        model_pred[: dspo_actionability.shape[0]].detach().float()
                                        - ref_pred[: dspo_actionability.shape[0]].float()
                                    )
                                )
                                dspo_reduce_dims = tuple(range(1, model_pred.ndim))
                                dspo_correction_to_data_ratio = (
                                    dspo_score_correction.square()
                                    .mean(dim=dspo_reduce_dims)
                                    .sqrt()
                                    / dspo_data_residual.square()
                                    .mean(dim=dspo_reduce_dims)
                                    .sqrt()
                                    .clamp_min(1e-12)
                                )
                                winner_anchor_loss = dspo_winner_score_anchor(
                                    model_prediction=model_pred,
                                    reference_prediction=ref_pred,
                                    target=target,
                                    preferred_probability=dspo_preferred_probability,
                                    correction_scale=args.dspo_score_correction_scale,
                                )
                            else:
                                winner_anchor_loss = winner_denoising_anchor(
                                    model_prediction=model_pred,
                                    target=target,
                                )
                            winner_anchor_output_gradient = loss_output_gradient(
                                winner_anchor_loss, model_pred
                            )
                            winner_anchor_output_gradient_norm = global_output_gradient_norm(
                                winner_anchor_output_gradient
                            )
                            winner_anchor_output_gradient_cosine = (
                                global_output_gradient_cosine(
                                    ratio_primary_output_gradient,
                                    winner_anchor_output_gradient,
                                    ratio_primary_output_gradient_norm,
                                    winner_anchor_output_gradient_norm,
                                )
                            )
                            winner_anchor_scale = auxiliary_output_gradient_scale(
                                primary_gradient_norm=ratio_primary_output_gradient_norm,
                                auxiliary_gradient_norm=winner_anchor_output_gradient_norm,
                                target_ratio=args.winner_anchor_output_gradient_ratio,
                                maximum_scale=(
                                    args.auxiliary_output_gradient_scale_max
                                ),
                            )

                        loss = (
                            ratio_policy_loss
                            + confidence_objective
                            + reference_anchor_scale * reference_anchor_loss
                            + winner_anchor_scale * winner_anchor_loss
                        )
                        ratio_mean_distance = reduce_latent(
                            (mean_ref.float() - mean_theta.float()).square(), args.ratio_reduction
                        ).sqrt()
                        ratio_distance_w, ratio_distance_l = ratio_mean_distance.chunk(2)
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / accumulation_window_size
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method == 'dpo':
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    model_mse_accumulated += avg_model_mse / accumulation_window_size
                    reference_mse_accumulated += avg_ref_mse / accumulation_window_size
                    implicit_acc_accumulated += avg_acc / accumulation_window_size
                    if args.preference_loss in RATIO_PREFERENCE_LOSSES:
                        posterior_variance = coefficients.variance.float().flatten(start_dim=1)[:, 0]
                        ratio_diagnostic_window.append(
                            {
                                "log_r": accelerator.gather(ratio_log_r.detach().float()).flatten(),
                                "raw_margin": accelerator.gather(
                                    ratio_raw_margin.detach().float()
                                ).flatten(),
                                "delta_w": accelerator.gather(ratio_delta_w.detach().float()).flatten(),
                                "delta_l": accelerator.gather(ratio_delta_l.detach().float()).flatten(),
                                "kl_w": accelerator.gather(ratio_kl_w.detach().float()).flatten(),
                                "kl_l": accelerator.gather(ratio_kl_l.detach().float()).flatten(),
                                "attached_kl": accelerator.gather(
                                    attached_reference_kl.detach().float()
                                ).flatten(),
                                "log_omega": accelerator.gather(
                                    ratio_log_omega.detach().float()
                                ).flatten(),
                                "unscaled_pair_loss": accelerator.gather(
                                    ratio_unscaled_per_pair_loss.detach().float()
                                ).flatten(),
                                "scaled_pair_loss": accelerator.gather(
                                    ratio_per_pair_loss.detach().float()
                                ).flatten(),
                                "policy_loss": accelerator.gather(
                                    ratio_policy_loss.detach().float().reshape(1)
                                ).flatten(),
                                "ratio_loss_scale": accelerator.gather(
                                    torch.as_tensor(
                                        args.ratio_loss_scale,
                                        device=loss.device,
                                        dtype=torch.float32,
                                    ).reshape(1)
                                ).flatten(),
                                "primary_output_gradient_norm": accelerator.gather(
                                    ratio_primary_output_gradient_norm.reshape(1)
                                ).flatten(),
                                "reference_anchor_loss": accelerator.gather(
                                    reference_anchor_loss.detach().float().reshape(1)
                                ).flatten(),
                                "reference_anchor_scale": accelerator.gather(
                                    reference_anchor_scale.detach().float().reshape(1)
                                ).flatten(),
                                "reference_anchor_output_gradient_norm": accelerator.gather(
                                    reference_anchor_output_gradient_norm.reshape(1)
                                ).flatten(),
                                "reference_anchor_output_gradient_cosine": accelerator.gather(
                                    reference_anchor_output_gradient_cosine.reshape(1)
                                ).flatten(),
                                "winner_anchor_loss": accelerator.gather(
                                    winner_anchor_loss.detach().float().reshape(1)
                                ).flatten(),
                                "winner_anchor_scale": accelerator.gather(
                                    winner_anchor_scale.detach().float().reshape(1)
                                ).flatten(),
                                "winner_anchor_output_gradient_norm": accelerator.gather(
                                    winner_anchor_output_gradient_norm.reshape(1)
                                ).flatten(),
                                "winner_anchor_output_gradient_cosine": accelerator.gather(
                                    winner_anchor_output_gradient_cosine.reshape(1)
                                ).flatten(),
                                "winner_model_mse": accelerator.gather(
                                    model_losses_w.detach().float()
                                ).flatten(),
                                "loser_model_mse": accelerator.gather(
                                    model_losses_l.detach().float()
                                ).flatten(),
                                "distance_w": accelerator.gather(
                                    ratio_distance_w.detach().float()
                                ).flatten(),
                                "distance_l": accelerator.gather(
                                    ratio_distance_l.detach().float()
                                ).flatten(),
                                "posterior_variance": accelerator.gather(
                                    posterior_variance.detach().float()
                                ).flatten(),
                                "exponent_clipped": accelerator.gather(
                                    ratio_exponent_clipped.detach().float()
                                ).flatten(),
                                **(
                                    {
                                        "step_confidence": accelerator.gather(
                                            step_confidence.detach().float()
                                        ).flatten(),
                                        "confidence_target": accelerator.gather(
                                            confidence_target.detach().float()
                                        ).flatten(),
                                        "policy_weight": accelerator.gather(
                                            step_aware_output.policy_weights.detach().float()
                                        ).flatten(),
                                        "raw_policy_weight": accelerator.gather(
                                            step_aware_output.raw_policy_weights.detach().float()
                                        ).flatten(),
                                        "policy_weight_normalizer": accelerator.gather(
                                            step_aware_output.policy_weight_normalizer.detach().float().reshape(1)
                                        ).flatten(),
                                        "confidence_loss": accelerator.gather(
                                            step_aware_output.confidence_loss.detach().float().reshape(1)
                                        ).flatten(),
                                        "weighted_policy_loss": accelerator.gather(
                                            step_aware_output.policy_loss.detach().float().reshape(1)
                                        ).flatten(),
                                        "pair_timestep": accelerator.gather(
                                            pair_timesteps.detach().float()
                                        ).flatten(),
                                    }
                                    if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES
                                    else {}
                                ),
                                **(
                                    {
                                        "lrm_teacher_probability": accelerator.gather(
                                            lrm_teacher_probability.detach().float()
                                        ).flatten(),
                                        "lrm_teacher_score_gap": accelerator.gather(
                                            lrm_teacher_score_gap.detach().float()
                                        ).flatten(),
                                    }
                                    if lrm_teacher_probability is not None
                                    else {}
                                ),
                                **(
                                    {
                                        "dspo_preferred_probability": accelerator.gather(
                                            dspo_preferred_probability.detach().float()
                                        ).flatten(),
                                        "dspo_preference_logit": accelerator.gather(
                                            dspo_preference_logit.detach().float()
                                        ).flatten(),
                                        "dspo_actionability": accelerator.gather(
                                            dspo_actionability.detach().float()
                                        ).flatten(),
                                        "dspo_residual_jacobian_scale": accelerator.gather(
                                            dspo_residual_jacobian_scale.detach().float()
                                        ).flatten(),
                                        "dspo_correction_to_data_ratio": accelerator.gather(
                                            dspo_correction_to_data_ratio.detach().float()
                                        ).flatten(),
                                    }
                                    if dspo_preferred_probability is not None
                                    else {}
                                ),
                            }
                        )

                        def concatenate_ratio_metric(name):
                            return torch.cat([item[name] for item in ratio_diagnostic_window])

                        gathered_log_r = concatenate_ratio_metric("log_r")
                        finite_log_r = gathered_log_r[torch.isfinite(gathered_log_r)]
                        if finite_log_r.numel() > 0:
                            finite_margin = -finite_log_r
                            ratio_log_r_stats = {
                                "ratio/log_r_mean": finite_log_r.mean().item(),
                                "ratio/log_r_std": finite_log_r.std(unbiased=False).item(),
                                "ratio/log_r_min": finite_log_r.min().item(),
                                "ratio/log_r_max": finite_log_r.max().item(),
                                "ratio/log_r_p01": torch.quantile(finite_log_r, 0.01).item(),
                                "ratio/log_r_p50": torch.quantile(finite_log_r, 0.50).item(),
                                "ratio/log_r_p99": torch.quantile(finite_log_r, 0.99).item(),
                                "ratio/preference_accuracy": (finite_log_r < 0).float().mean().item(),
                                "ratio/margin_mean": finite_margin.mean().item(),
                                "ratio/margin_std": finite_margin.std(unbiased=False).item(),
                                "ratio/margin_p01": torch.quantile(finite_margin, 0.01).item(),
                                "ratio/margin_p50": torch.quantile(finite_margin, 0.50).item(),
                                "ratio/margin_p99": torch.quantile(finite_margin, 0.99).item(),
                            }
                        else:
                            ratio_log_r_stats = {
                                key: float("nan")
                                for key in (
                                    "ratio/log_r_mean",
                                    "ratio/log_r_std",
                                    "ratio/log_r_min",
                                    "ratio/log_r_max",
                                    "ratio/log_r_p01",
                                    "ratio/log_r_p50",
                                    "ratio/log_r_p99",
                                    "ratio/preference_accuracy",
                                    "ratio/margin_mean",
                                    "ratio/margin_std",
                                    "ratio/margin_p01",
                                    "ratio/margin_p50",
                                    "ratio/margin_p99",
                                )
                            }

                        ratio_logging = {
                            **ratio_log_r_stats,
                            "ratio/raw_margin_mean": concatenate_ratio_metric(
                                "raw_margin"
                            ).mean().item(),
                            "ratio/raw_margin_std": concatenate_ratio_metric(
                                "raw_margin"
                            ).std(unbiased=False).item(),
                            "ratio/delta_w_mean": concatenate_ratio_metric("delta_w").mean().item(),
                            "ratio/delta_l_mean": concatenate_ratio_metric("delta_l").mean().item(),
                            "ratio/kl_w_mean": concatenate_ratio_metric("kl_w").mean().item(),
                            "ratio/kl_l_mean": concatenate_ratio_metric("kl_l").mean().item(),
                            "ratio/attached_reference_kl_mean": concatenate_ratio_metric(
                                "attached_kl"
                            ).mean().item(),
                            "ratio/log_omega_mean": concatenate_ratio_metric("log_omega").mean().item(),
                            "ratio/unscaled_pair_loss_mean": concatenate_ratio_metric(
                                "unscaled_pair_loss"
                            ).mean().item(),
                            "ratio/scaled_pair_loss_mean": concatenate_ratio_metric(
                                "scaled_pair_loss"
                            ).mean().item(),
                            "ratio/policy_loss": concatenate_ratio_metric(
                                "policy_loss"
                            ).mean().item(),
                            "ratio/loss_scale": concatenate_ratio_metric(
                                "ratio_loss_scale"
                            ).mean().item(),
                            "ratio/primary_output_gradient_norm": concatenate_ratio_metric(
                                "primary_output_gradient_norm"
                            ).mean().item(),
                            "anchor/reference_loss": concatenate_ratio_metric(
                                "reference_anchor_loss"
                            ).mean().item(),
                            "anchor/reference_scale": concatenate_ratio_metric(
                                "reference_anchor_scale"
                            ).mean().item(),
                            "anchor/reference_output_gradient_norm": concatenate_ratio_metric(
                                "reference_anchor_output_gradient_norm"
                            ).mean().item(),
                            "anchor/reference_output_gradient_cosine": concatenate_ratio_metric(
                                "reference_anchor_output_gradient_cosine"
                            ).mean().item(),
                            "anchor/winner_loss": concatenate_ratio_metric(
                                "winner_anchor_loss"
                            ).mean().item(),
                            "anchor/winner_scale": concatenate_ratio_metric(
                                "winner_anchor_scale"
                            ).mean().item(),
                            "anchor/winner_output_gradient_norm": concatenate_ratio_metric(
                                "winner_anchor_output_gradient_norm"
                            ).mean().item(),
                            "anchor/winner_output_gradient_cosine": concatenate_ratio_metric(
                                "winner_anchor_output_gradient_cosine"
                            ).mean().item(),
                            "quality/winner_model_mse": concatenate_ratio_metric(
                                "winner_model_mse"
                            ).mean().item(),
                            "quality/loser_model_mse": concatenate_ratio_metric(
                                "loser_model_mse"
                            ).mean().item(),
                            "ratio/current_ref_mean_distance_w": concatenate_ratio_metric(
                                "distance_w"
                            ).mean().item(),
                            "ratio/current_ref_mean_distance_l": concatenate_ratio_metric(
                                "distance_l"
                            ).mean().item(),
                            "ratio/posterior_variance_mean": concatenate_ratio_metric(
                                "posterior_variance"
                            ).mean().item(),
                            "ratio/nonfinite_fraction": (
                                (~torch.isfinite(gathered_log_r)).float().mean().item()
                            ),
                            "ratio/exponent_clip_fraction": concatenate_ratio_metric(
                                "exponent_clipped"
                            ).mean().item(),
                        }
                        primary_output_norms = concatenate_ratio_metric(
                            "primary_output_gradient_norm"
                        )
                        for anchor_name in ("reference", "winner"):
                            anchor_scales = concatenate_ratio_metric(
                                f"{anchor_name}_anchor_scale"
                            )
                            anchor_output_norms = concatenate_ratio_metric(
                                f"{anchor_name}_anchor_output_gradient_norm"
                            )
                            achieved_ratios = torch.where(
                                primary_output_norms > 0,
                                anchor_scales
                                * anchor_output_norms
                                / primary_output_norms.clamp_min(1e-12),
                                torch.zeros_like(primary_output_norms),
                            )
                            ratio_logging[
                                f"anchor/{anchor_name}_achieved_output_gradient_ratio"
                            ] = achieved_ratios.mean().item()
                            ratio_logging[
                                f"anchor/{anchor_name}_scale_cap_fraction"
                            ] = (
                                anchor_scales
                                >= 0.999999
                                * float(args.auxiliary_output_gradient_scale_max)
                            ).float().mean().item()
                        if args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES:
                            gathered_confidence = concatenate_ratio_metric(
                                "step_confidence"
                            )
                            gathered_target = concatenate_ratio_metric("confidence_target")
                            normalized_timestep = concatenate_ratio_metric(
                                "pair_timestep"
                            ) / float(num_train_timesteps - 1)
                            ratio_logging.update(
                                {
                                    "step_aware/confidence_mean": gathered_confidence.mean().item(),
                                    "step_aware/confidence_std": gathered_confidence.std(
                                        unbiased=False
                                    ).item(),
                                    "step_aware/target_mean": gathered_target.mean().item(),
                                    "step_aware/policy_weight_mean": concatenate_ratio_metric(
                                        "policy_weight"
                                    ).mean().item(),
                                    "step_aware/raw_policy_weight_mean": concatenate_ratio_metric(
                                        "raw_policy_weight"
                                    ).mean().item(),
                                    "step_aware/policy_weight_normalizer": concatenate_ratio_metric(
                                        "policy_weight_normalizer"
                                    ).mean().item(),
                                    "step_aware/confidence_bce": concatenate_ratio_metric(
                                        "confidence_loss"
                                    ).mean().item(),
                                    "step_aware/weighted_policy_loss": concatenate_ratio_metric(
                                        "weighted_policy_loss"
                                    ).mean().item(),
                                    "step_aware/calibration_mae": (
                                        gathered_confidence - gathered_target
                                    ).abs().mean().item(),
                                }
                            )
                            timestep_bins = {
                                "t_0_25": normalized_timestep < 0.25,
                                "t_25_50": (normalized_timestep >= 0.25)
                                & (normalized_timestep < 0.50),
                                "t_50_75": (normalized_timestep >= 0.50)
                                & (normalized_timestep < 0.75),
                                "t_75_100": normalized_timestep >= 0.75,
                            }
                            for bin_name, bin_mask in timestep_bins.items():
                                ratio_logging[
                                    f"step_aware/{bin_name}_fraction"
                                ] = bin_mask.float().mean().item()
                                if torch.any(bin_mask):
                                    ratio_logging[
                                        f"step_aware/confidence_{bin_name}_mean"
                                    ] = gathered_confidence[bin_mask].mean().item()
                                    ratio_logging[
                                        f"step_aware/target_{bin_name}_mean"
                                    ] = gathered_target[bin_mask].mean().item()
                                else:
                                    ratio_logging[
                                        f"step_aware/confidence_{bin_name}_mean"
                                    ] = float("nan")
                                    ratio_logging[
                                        f"step_aware/target_{bin_name}_mean"
                                    ] = float("nan")
                            if args.confidence_target_source == "precomputed_lrm":
                                gathered_lrm_probability = concatenate_ratio_metric(
                                    "lrm_teacher_probability"
                                )
                                gathered_lrm_score_gap = concatenate_ratio_metric(
                                    "lrm_teacher_score_gap"
                                )
                                ratio_logging.update(
                                    {
                                        "lrm/preferred_probability_mean": gathered_lrm_probability.mean().item(),
                                        "lrm/score_gap_mean": gathered_lrm_score_gap.mean().item(),
                                        "lrm/score_gap_p05": torch.quantile(gathered_lrm_score_gap, 0.05).item(),
                                        "lrm/score_gap_median": torch.quantile(gathered_lrm_score_gap, 0.50).item(),
                                        "lrm/score_gap_p95": torch.quantile(gathered_lrm_score_gap, 0.95).item(),
                                        "lrm/human_agreement_rate": (
                                            gathered_lrm_probability > 0.5
                                        ).float().mean().item(),
                                        "lrm/brier_against_human_winner": (
                                            gathered_lrm_probability - 1.0
                                        ).square().mean().item(),
                                    }
                                )
                                for bin_name, bin_mask in timestep_bins.items():
                                    ratio_logging[
                                        f"lrm/preferred_probability_{bin_name}_mean"
                                    ] = (
                                        gathered_lrm_probability[bin_mask].mean().item()
                                        if torch.any(bin_mask)
                                        else float("nan")
                                    )
                        if args.winner_anchor_type == "dspo_score":
                            gathered_dspo_probability = concatenate_ratio_metric(
                                "dspo_preferred_probability"
                            )
                            gathered_dspo_logit = concatenate_ratio_metric(
                                "dspo_preference_logit"
                            )
                            gathered_dspo_actionability = concatenate_ratio_metric(
                                "dspo_actionability"
                            )
                            gathered_dspo_jacobian = concatenate_ratio_metric(
                                "dspo_residual_jacobian_scale"
                            )
                            gathered_dspo_correction_ratio = concatenate_ratio_metric(
                                "dspo_correction_to_data_ratio"
                            )
                            ratio_logging.update(
                                {
                                    "dspo/preferred_probability_mean": gathered_dspo_probability.mean().item(),
                                    "dspo/logit_mean": gathered_dspo_logit.mean().item(),
                                    "dspo/logit_p05": torch.quantile(gathered_dspo_logit, 0.05).item(),
                                    "dspo/logit_p95": torch.quantile(gathered_dspo_logit, 0.95).item(),
                                    "dspo/actionability_mean": gathered_dspo_actionability.mean().item(),
                                    "dspo/residual_jacobian_scale_mean": gathered_dspo_jacobian.mean().item(),
                                    "dspo/residual_jacobian_scale_min": gathered_dspo_jacobian.min().item(),
                                    "dspo/correction_to_data_ratio_mean": gathered_dspo_correction_ratio.mean().item(),
                                    "dspo/correction_to_data_ratio_p95": torch.quantile(gathered_dspo_correction_ratio, 0.95).item(),
                                }
                            )
                        if args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES and finite_log_r.numel() > 0:
                            ms_margin_gradient = mode_seeking_margin_gradient_magnitude(
                                finite_log_r,
                                kappa=args.mode_seeking_kappa,
                                max_exp_argument=args.max_exp_argument,
                            )
                            dpo_margin_gradient = 2.0 * torch.sigmoid(finite_log_r)
                            ratio_logging.update(
                                {
                                    "ratio/ms_margin_gradient_mean": ms_margin_gradient.mean().item(),
                                    "ratio/ms_margin_gradient_p95": torch.quantile(ms_margin_gradient, 0.95).item(),
                                    "ratio/ms_margin_gradient_p99": torch.quantile(ms_margin_gradient, 0.99).item(),
                                    "ratio/ms_margin_gradient_max": ms_margin_gradient.max().item(),
                                    "ratio/dpo_margin_gradient_mean": dpo_margin_gradient.mean().item(),
                                }
                            )

                            ratio_bins = {
                                "r_lt_0_1": finite_log_r < math.log(0.1),
                                "r_0_1_to_1": (finite_log_r >= math.log(0.1)) & (finite_log_r < 0),
                                "r_ge_1": finite_log_r >= 0,
                            }
                            for bin_name, bin_mask in ratio_bins.items():
                                ratio_logging[f"ratio/{bin_name}_fraction"] = bin_mask.float().mean().item()
                                if torch.any(bin_mask):
                                    ratio_logging[
                                        f"ratio/ms_margin_gradient_{bin_name}_mean"
                                    ] = ms_margin_gradient[bin_mask].mean().item()
                                    ratio_logging[
                                        f"ratio/dpo_margin_gradient_{bin_name}_mean"
                                    ] = dpo_margin_gradient[bin_mask].mean().item()
                                else:
                                    ratio_logging[
                                        f"ratio/ms_margin_gradient_{bin_name}_mean"
                                    ] = float("nan")
                                    ratio_logging[
                                        f"ratio/dpo_margin_gradient_{bin_name}_mean"
                                    ] = float("nan")

                # Backpropagate
                # Accelerate 0.20 always divides by the configured accumulation
                # count. Correct the final partial window so it has the same
                # mean-loss normalization as a full window.
                backward_loss = loss * (
                    args.gradient_accumulation_steps / accumulation_window_size
                )
                accelerator.backward(backward_loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        gradient_norm = accelerator.clip_grad_norm_(
                            trainable_parameters, args.max_grad_norm
                        )
                    else:
                        parameter_grad_norms = [
                            parameter.grad.detach().float().norm(2)
                            for parameter in trainable_parameters
                            if parameter.grad is not None
                        ]
                        gradient_norm = (
                            torch.linalg.vector_norm(torch.stack(parameter_grad_norms))
                            if parameter_grad_norms
                            else torch.zeros((), device=loss.device)
                        )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method == 'dpo':
                    accelerator.log(
                        {
                            "model_mse_unaccumulated": avg_model_mse,
                            "model_mse": model_mse_accumulated,
                        },
                        step=global_step,
                    )
                    accelerator.log(
                        {
                            "ref_mse_unaccumulated": avg_ref_mse,
                            "reference_mse": reference_mse_accumulated,
                        },
                        step=global_step,
                    )
                    accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                    if args.preference_loss in RATIO_PREFERENCE_LOSSES:
                        avg_gradient_norm = accelerator.gather(
                            torch.as_tensor(gradient_norm, device=loss.device, dtype=torch.float32).reshape(1)
                        ).mean().item()
                        accelerator.log(
                            {**ratio_logging, "train/gradient_norm": avg_gradient_norm}, step=global_step
                        )
                train_loss = 0.0
                implicit_acc_accumulated = 0.0
                model_mse_accumulated = 0.0
                reference_mse_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                name for name in os.listdir(args.output_dir)
                                if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()
                            ]
                            checkpoints.sort(key=lambda name: int(name.split("-")[-1]))
                            while len(checkpoints) >= args.checkpoints_total_limit:
                                remove_path = os.path.join(args.output_dir, checkpoints.pop(0))
                                logger.info(f"Removing old checkpoint {remove_path}")
                                shutil.rmtree(remove_path)
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        if args.eval_checkpoint_dir:
                            archive_root = Path(args.eval_checkpoint_dir)
                            archive_root.mkdir(parents=True, exist_ok=True)
                            archive_path = archive_root / f"checkpoint-{global_step}"
                            archive_tmp = archive_root / f".checkpoint-{global_step}.tmp"
                            expected_weights = (
                                archive_path / "unet" / "diffusion_pytorch_model.safetensors"
                            )
                            if not expected_weights.is_file():
                                if archive_path.exists():
                                    shutil.rmtree(archive_path)
                                if archive_tmp.exists():
                                    shutil.rmtree(archive_tmp)
                                archive_tmp.mkdir(parents=True)

                                def link_or_copy(source, destination):
                                    try:
                                        os.link(source, destination)
                                    except OSError:
                                        shutil.copy2(source, destination)

                                shutil.copytree(
                                    Path(save_path) / "unet",
                                    archive_tmp / "unet",
                                    copy_function=link_or_copy,
                                )
                                for filename in (
                                    "step_confidence_head.pt",
                                    "ratio_diffusion_config.json",
                                ):
                                    source = Path(save_path) / filename
                                    if source.is_file():
                                        link_or_copy(source, archive_tmp / filename)
                                os.replace(archive_tmp, archive_path)
                            logger.info(f"Archived evaluation snapshot to {archive_path}")
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if confidence_head is not None:
                logs["confidence_lr"] = lr_scheduler.get_last_lr()[1]
            if args.train_method == 'dpo':
                logs["implicit_acc"] = avg_acc
                if args.preference_loss in RATIO_PREFERENCE_LOSSES:
                    logs["log_r"] = ratio_logging["ratio/log_r_mean"]
                    logs["clip"] = ratio_logging["ratio/exponent_clip_fraction"]
            progress_bar.set_postfix(**logs)
            if accelerator.sync_gradients and args.preference_loss in RATIO_PREFERENCE_LOSSES:
                ratio_diagnostic_window.clear()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if confidence_head is not None:
            confidence_head = accelerator.unwrap_model(confidence_head)
        if args.sdxl:
            # Serialize pipeline.
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path, unet=unet, vae=vae, revision=args.revision, torch_dtype=weight_dtype
            )
            pipeline.save_pretrained(args.output_dir)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)
        if ratio_config is not None:
            with open(
                os.path.join(args.output_dir, "ratio_diffusion_config.json"),
                "w",
                encoding="utf-8",
            ) as config_file:
                json.dump(ratio_config, config_file, indent=2, sort_keys=True)
        if confidence_head is not None:
            torch.save(
                confidence_head.state_dict(),
                os.path.join(args.output_dir, "step_confidence_head.pt"),
            )


    accelerator.end_training()


if __name__ == "__main__":
    main()
