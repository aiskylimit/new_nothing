"""Validation and response alignment helpers for finetune_v2."""

import math

import torch

from .batching import pack_trajectories


def validate_mode_args(args):
    adaptive = getattr(args, "adaptive_on_policy", False)
    if adaptive:
        from .adaptive import AdaptiveConfig
        config = AdaptiveConfig.from_args(args)
        if args.student_gen or args.distill_mode not in (None, "off_policy") or args.type not in (None, "kd"):
            raise ValueError("--adaptive-on-policy requires --type kd and no fixed ON/PRIV mode or --student-gen")
        if not args.teacher_model_path:
            raise ValueError("--adaptive-on-policy requires --teacher-model-path")
        if args.do_train and (not args.eval_interval or args.eval_interval < -1):
            raise ValueError("--adaptive-on-policy requires positive --eval-interval (or -1 for each epoch)")
        if config.progress_signal in ("metric", "either") and not getattr(args, "eval_gen", False):
            raise ValueError("--progress-signal metric/either requires --eval-gen")
    legacy_type = args.type or "kd"
    if "adaptive" in legacy_type or "mixed" in legacy_type:
        raise ValueError("Use --distill-mode; "
                         "adaptive/mixed routing is not supported by finetune_v2")
    if legacy_type == "rvd":
        raise ValueError("Use finetune.py for --type rvd; finetune_v2 uses --type kd")
    if args.student_gen:
        if args.distill_mode not in (None, "on_policy"):
            raise ValueError("--student-gen conflicts with --distill-mode")
        args.distill_mode = "on_policy"
    args.distill_mode = args.distill_mode or (
        legacy_type if legacy_type in ("on_policy", "privileged") else "off_policy"
    )
    if getattr(args, "privileged_data_path", None):
        if args.distill_mode != "privileged" and not adaptive:
            raise ValueError("--privileged-data-path requires privileged mode or --adaptive-on-policy")
        if args.privileged_trajectory == "student":
            raise ValueError("--privileged-data-path uses dataset responses; remove --privileged-trajectory student")
    if "off_policy" in legacy_type:
        args.off_policy_geometry = True
    if args.kd_loss is None:
        args.kd_loss = next((name for name in ("sfkl", "srkl", "jsd", "tvd", "fkl", "rkl")
                             if name in legacy_type), None)
        if args.kd_loss is None:
            if legacy_type not in ("kd", "lm", "off_policy", "on_policy", "privileged"):
                raise ValueError("Specify --kd-loss for this legacy --type")
            args.kd_loss = "fkl"
    if args.kd_ratio is None:
        args.kd_ratio = 1.0
    if not math.isfinite(args.kd_ratio) or not 0 <= args.kd_ratio <= 1:
        raise ValueError("--kd-ratio must be in [0, 1]")
    if not math.isfinite(args.skew_alpha) or not 0 <= args.skew_alpha <= 1:
        raise ValueError("--skew-alpha must be in [0, 1]")
    args.distill_top_k = getattr(args, "distill_top_k", 32)
    args.distill_temperature = getattr(args, "distill_temperature", 1.0)
    if args.distill_top_k < 2:
        raise ValueError("--distill-top-k must be at least 2")
    if not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
        raise ValueError("--distill-temperature must be finite and positive")
    uses_generation = args.distill_mode == "on_policy"
    if (args.distill_mode == "privileged" or adaptive) and args.privileged_trajectory != "canonical":
        raise ValueError("Privileged distillation uses dataset responses; use --privileged-trajectory canonical")
    if args.privileged_trajectory != "canonical" and args.distill_mode != "privileged" and not adaptive:
        raise ValueError("--privileged-trajectory applies only to privileged mode")
    if args.off_policy_geometry and args.distill_mode != "off_policy":
        raise ValueError("Geometry is supported only for off_policy")
    if any(not math.isfinite(w) or w < 0 for w in (args.mag_weight, args.gram_weight)):
        raise ValueError("Geometry weights must be finite and nonnegative")
    if not math.isfinite(args.eps) or args.eps <= 0:
        raise ValueError("--eps must be finite and positive")
    if args.do_train and not args.teacher_model_path and legacy_type != "lm":
        raise ValueError("Distillation requires --teacher-model-path")
    if legacy_type == "lm" and (uses_generation or args.distill_mode == "privileged" or args.disable_lm_loss):
        raise ValueError("--type lm requires canonical LM supervision")
    if not 0 < args.max_prompt_length < args.max_length:
        raise ValueError("Require 0 < --max-prompt-length < --max-length")
    if (args.distill_mode == "privileged" or adaptive) and not 0 < args.t_max_prompt_length < args.t_max_length:
        raise ValueError("Require 0 < --t-max-prompt-length < --t-max-length")
    if args.do_train and (args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.log_interval < 1):
        raise ValueError("Batch size, gradient accumulation and log interval must be positive")


def require_shared_vocabulary(student_tokenizer, teacher_tokenizer):
    if (student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
            or student_tokenizer.eos_token_id != teacher_tokenizer.eos_token_id):
        raise ValueError("Token-level KD requires matching vocabulary/token IDs and EOS; "
                         "different tokenizers need an explicit vocabulary mapping")


def align_response_logits(student_logits, student_labels, teacher_logits, teacher_labels):
    """[B,Ls,V], [B,Ls], [B,Lt,V], [B,Lt] -> [N,V], [N,V], [N].

    Labels already contain next-token targets. Compare prediction positions by
    response index, including EOS. Reject mismatches rather than silently crop.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("Expected student/teacher logits [B, L, V]")
    if student_logits.shape[:2] != student_labels.shape or teacher_logits.shape[:2] != teacher_labels.shape:
        raise ValueError("Logit sequence dimensions must match labels")
    if (student_logits.shape[0] != teacher_logits.shape[0]
            or student_logits.shape[-1] != teacher_logits.shape[-1]):
        raise ValueError("Batch size and vocabulary dimensions must match")
    sm, tm = student_labels != -100, teacher_labels != -100
    if not torch.equal(sm.sum(-1), tm.sum(-1)):
        raise ValueError("Student/teacher response lengths differ")
    if not torch.equal(student_labels[sm], teacher_labels[tm]):
        raise ValueError("Student/teacher response token IDs differ")
    return student_logits[sm], teacher_logits[tm], student_labels[sm]


def prepare_privileged_batches(args, tokenizer, student_batch, metadata):
    """Apply both models' budgets while keeping their response labels identical."""
    if not 0 < args.t_max_prompt_length < args.t_max_length:
        raise ValueError("Require 0 < --t-max-prompt-length < --t-max-length")
    if not 0 < args.max_prompt_length < args.max_length:
        raise ValueError("Require 0 < --max-prompt-length < --max-length")
    student_prompts, teacher_prompts, responses = [], [], []
    for ids, labels, teacher_prompt in zip(student_batch["input_ids"], metadata["label"],
                                            metadata["privileged_prompt_ids"]):
        positions = (labels != -100).nonzero(as_tuple=True)[0]
        if not positions.numel():
            raise ValueError("Privileged distillation requires nonempty student response labels")
        prompt_length = int(positions[0]) + 1
        student_prompt = ids[:prompt_length][-args.max_prompt_length:]
        student_prompts.append(student_prompt)
        # Reserve the shared response first. A smaller teacher budget trims its
        # prompt/context before sacrificing any of the student's response labels.
        # Only an answer that cannot fit with even one teacher prompt token is
        # shortened, using the same prefix (including EOS if it fits) for both.
        response_budget = min(args.max_length - len(student_prompt),
                              args.t_max_length - 1)
        response = labels[positions][:response_budget]
        teacher_prompt_budget = min(args.t_max_prompt_length, args.t_max_length - len(response))
        teacher_prompts.append(teacher_prompt[-teacher_prompt_budget:])
        responses.append(response)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = student_batch["input_ids"].device
    student, student_metadata = pack_trajectories(
        student_prompts, responses, pad_id, args.model_type, args.max_length, device,
    )
    teacher, teacher_metadata = pack_trajectories(
        teacher_prompts, responses, pad_id, args.teacher_model_type or args.model_type,
        args.t_max_length, device,
    )
    return student, {**metadata, **student_metadata}, teacher, teacher_metadata
