"""Bộ máy step-weighted SFT dùng chung cho Vanilla và Entropy Selection.

Vanilla = đúng bộ máy này với ``w_i ≡ 1``. Nhờ vậy ablation ``uniform`` của §9
**bằng đúng** Vanilla SFT chứ không chỉ "gần bằng" — nếu hai con số lệch nhau
thì lỗi nằm ở weighting/gate chứ không phải ở khác biệt cài đặt loss (§14).

Segment của một mẫu = các step trong phần thinking + một segment cuối cho phần
sau ``</think>`` (luôn có trọng số 1, không tham gia tính U/V).
"""

from __future__ import annotations

import os
import time
from typing import Callable, List, Optional, Tuple

import torch

from core.data_utils import JsonlLogger, StepDataset, TrainExample, set_seed
from core.losses import (
    step_nll_from_logprobs,
    step_token_counts,
    token_logprobs,
    weighted_step_loss,
)
from core.sed import k2_step_loss, teacher_sed_logprobs
from core.model_utils import (
    attach_single_lora,
    load_base_model,
    load_tokenizer,
    make_lora_config,
)
from train.common import (
    Accumulator,
    build_optimizer,
    clip_and_step,
    cosine_schedule_with_warmup,
    resolve_device,
    save_run_config,
    total_optimizer_steps,
)

Span = Tuple[int, int]
WeightProvider = Callable[[TrainExample, List[Span], torch.device], Optional[torch.Tensor]]


def build_segments(example: TrainExample, include_answer: bool = True) -> List[Span]:
    """Step spans + segment câu trả lời cuối (nếu có token)."""
    spans = list(example.step_spans)
    if include_answer:
        answer_start, answer_end = example.answer_span
        if answer_end > answer_start:
            spans.append((answer_start, answer_end))
    return spans


def n_thinking_steps(example: TrainExample) -> int:
    return len(example.step_spans)


def sed_config_from_args(args) -> Optional[dict]:
    """Đọc cấu hình Self-Exploratory Distillation (CurioSFT) từ ``args``.

    Trả ``None`` nếu ``--use_sed`` không bật ⇒ Vanilla/Entropy chạy y như cũ.
    """
    if not getattr(args, "use_sed", False):
        return None
    return {
        "alpha": float(getattr(args, "sed_alpha", 1.0)),
        "adaptive_tau": bool(getattr(args, "sed_adaptive_tau", True)),
        "fixed_tau": float(getattr(args, "sed_fixed_tau", 1.3)),
        "disable_adapter": getattr(args, "sed_teacher", "frozen_base") == "frozen_base",
        "chunk_size": int(getattr(args, "sed_chunk_size", 256)),
        "weighted": bool(getattr(args, "sed_weighted", True)),
    }


def pad_weights_for_answer(
    weights: torch.Tensor, n_segments: int, answer_weight: float = 1.0
) -> torch.Tensor:
    """Nối trọng số 1 cho segment answer để khớp số segment."""
    if weights.numel() == n_segments:
        return weights
    if weights.numel() != n_segments - 1:
        raise ValueError(f"weights {weights.numel()} không khớp {n_segments} segment")
    tail = weights.new_full((1,), answer_weight)
    return torch.cat([weights, tail], dim=0)


def run_step_weighted_sft(
    args,
    weight_provider: Optional[WeightProvider] = None,
    *,
    tag: str = "sft",
    setup_hook: Optional[Callable] = None,
) -> str:
    """Train một LoRA adapter với loss step-weighted.

    ``weight_provider(example, spans, device)`` trả ``w`` độ dài bằng số step
    thinking (hoặc bằng số segment) — trả ``None`` nghĩa là w ≡ 1 (Vanilla).
    """
    device = resolve_device(args.device)
    set_seed(args.seed)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    base = load_base_model(
        args.model_name,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    lora_config = make_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = attach_single_lora(base, lora_config)
    model.train()

    if setup_hook is not None:
        setup_hook(model, tokenizer, device)

    dataset = StepDataset(
        args.train_file,
        tokenizer,
        cutoff_len=args.cutoff_len,
        segmentation_mode=args.segmentation_mode,
        limit=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{args.train_file} rỗng")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, args.lr, args.weight_decay)
    total_steps = total_optimizer_steps(len(dataset), args.epochs, args.global_batch)
    scheduler = cosine_schedule_with_warmup(optimizer, total_steps, args.warmup_ratio)
    logger = JsonlLogger(out_dir)
    save_run_config(out_dir, args, {"tag": tag, "n_examples": len(dataset),
                                    "total_optimizer_steps": total_steps})

    sed = sed_config_from_args(args)
    if sed is not None:
        print(f"[{tag}] SED bật: alpha={sed['alpha']} adaptive_tau={sed['adaptive_tau']} "
              f"teacher={'frozen_base' if sed['disable_adapter'] else 'detached_student'} "
              f"weighted={sed['weighted']}")

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    accumulator = Accumulator()
    micro, opt_step = 0, 0
    optimizer.zero_grad(set_to_none=True)
    run_t0 = time.time()
    step_t0 = run_t0

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    for epoch in range(args.epochs):
        for index in dataset.iter_epoch(args.seed, epoch):
            example = dataset[index]
            spans = build_segments(example)
            if not spans:
                continue

            ex_t0 = time.time()
            input_ids = example.input_ids.unsqueeze(0).to(device)
            out = model(input_ids=input_ids, use_cache=False)
            student_lp = token_logprobs(out.logits, input_ids)
            losses = step_nll_from_logprobs(student_lp, spans)
            lengths = step_token_counts(spans)

            weights = None
            if weight_provider is not None:
                weights = weight_provider(example, spans, device)
                if weights is not None:
                    weights = pad_weights_for_answer(weights.to(device), len(spans))

            k2_mean = None
            teacher_sec = None
            if sed is None:
                loss = weighted_step_loss(losses, lengths, weights, args.loss_reduction)
            else:
                _sync(); t_teach = time.time()
                teacher_lp = teacher_sed_logprobs(
                    model, input_ids,
                    adaptive_tau=sed["adaptive_tau"], fixed_tau=sed["fixed_tau"],
                    disable_adapter=sed["disable_adapter"], chunk_size=sed["chunk_size"],
                )
                _sync(); teacher_sec = time.time() - t_teach
                k2 = k2_step_loss(student_lp, teacher_lp, spans)  # (n_seg,)
                k2_mean = float(k2.detach().mean())
                if sed["weighted"]:
                    loss = weighted_step_loss(
                        losses + sed["alpha"] * k2, lengths, weights, args.loss_reduction
                    )
                else:
                    loss = weighted_step_loss(losses, lengths, weights, args.loss_reduction) \
                        + sed["alpha"] * weighted_step_loss(k2, lengths, None, args.loss_reduction)

            (loss / args.global_batch).backward()
            _sync()

            accumulator.add(
                loss=float(loss.detach()),
                mean_step_nll=float(losses.detach().mean()),
                mean_k2=k2_mean,
                n_steps=len(spans),
                seq_tokens=int(input_ids.shape[1]),
                example_sec=time.time() - ex_t0,
                teacher_sec=teacher_sec,
                mean_weight=float(weights.mean()) if weights is not None else 1.0,
            )
            del out, losses, student_lp

            micro += 1
            if micro % args.global_batch:
                continue

            grad_norm = clip_and_step(params, optimizer, scheduler, args.max_grad_norm)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == 1 or args.dry_run:
                now = time.time()
                means = accumulator.mean()
                record = {
                    "tag": tag, "epoch": epoch, "step": opt_step,
                    "total_steps": total_steps,
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "grad_norm": round(grad_norm, 4),
                    "sec_per_opt_step": round(now - step_t0, 2),
                    "elapsed_min": round((now - run_t0) / 60, 2),
                    "eta_min": round((now - run_t0) / 60 / opt_step * (total_steps - opt_step), 1),
                    "seq_tokens_mean": round(means.get("seq_tokens", 0), 1),
                    "example_sec_mean": round(means.get("example_sec", 0), 3),
                    "teacher_sec_mean": round(means.get("teacher_sec", 0), 3) if sed is not None else None,
                    "cuda_max_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if is_cuda else None,
                    **{k: round(v, 5) for k, v in means.items()
                       if k in ("loss", "mean_step_nll", "mean_k2", "mean_weight", "n_steps")},
                }
                logger.log(record)
                print(f"[{tag}] step {opt_step}/{total_steps} "
                      f"loss={record.get('loss', float('nan')):.4f} "
                      f"k2={record.get('mean_k2')} "
                      f"w̄={record.get('mean_weight', 1.0):.3f} "
                      f"| {record['sec_per_opt_step']}s/step "
                      f"teach={record.get('teacher_sec_mean')}s "
                      f"mem={record.get('cuda_max_gb')}GB "
                      f"elapsed={record['elapsed_min']}m eta={record['eta_min']}m",
                      flush=True)
                step_t0 = now
            accumulator.reset()

            if args.save_every and opt_step % args.save_every == 0:
                model.save_pretrained(os.path.join(out_dir, f"checkpoint-{opt_step}"))
            if args.dry_run and opt_step >= args.dry_run:
                print(f"[{tag}] dry_run đạt {opt_step} step — dừng")
                model.save_pretrained(out_dir)
                logger.close()
                return out_dir

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.close()
    print(f"[{tag}] xong -> {out_dir}")
    return out_dir
