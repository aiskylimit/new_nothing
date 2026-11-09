"""CORED data-parallel trainer, launched with torchrun (one replica per GPU).

Each replica contains the whole council, processes different examples and sums
only LoRA gradients once per global batch. Supports the main svgd_func recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from core.data_utils import StepDataset, set_seed, track_cli_provided, write_json
from core.efficient_loss import step_token_weights, weighted_nll_kl
from core.local_assets import local_adapter_path, local_model_path
from core.model_utils import (attach_experts, load_base_model, load_tokenizer,
                             lora_parameters, make_lora_config, set_lora_trainable)
from core.rep_loss import lora_diversity_loss
from proposal_v2.svgd import svgd_func_repulsion
from train.common import add_common_args, cosine_schedule_with_warmup, save_run_config
from train.parallel import (ParallelContext, backward_with_retry, global_windows,
                            pack_microbatches, restore_rng, rng_state, sync_gradients)
from train.phase2_distill import load_cached_w, segment_weights
from train.sft_engine import build_segments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--stage", choices=["phase1", "distill"], required=True)
    parser.add_argument("--micro_batch_size", type=int, default=11)
    parser.add_argument("--max_batch_tokens", type=int, default=32768)
    parser.add_argument("--loss_chunk", type=int, default=256)
    parser.add_argument("--attn_implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--p_drop", type=float, default=0.1)
    parser.add_argument("--lambda_rep", type=float, default=0.01)
    parser.add_argument("--diversity", choices=["svgd_func", "none"], default="svgd_func")
    parser.add_argument("--warmup_div_epochs", type=int, default=1)
    parser.add_argument("--tr_off_k", type=float, default=1.0)
    parser.add_argument("--svgd_sigma", type=float, default=0.0)
    parser.add_argument("--dpp_max_tokens", type=int, default=256)
    parser.add_argument("--keystep_dir", default=None)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--teacher_adapter", default=None)
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--uniform_weights", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Extra resume checkpoints per N updates; epoch checkpoints are default")
    parser.add_argument("--cpu_float32", action="store_true", help="CPU integration tests only")
    parser.set_defaults(model_name="models/Qwen2.5-7B-Instruct", global_batch=33,
                        cutoff_len=16384, loss_reduction="step_mean", out_dir="outputs/b200/phase1")
    args = track_cli_provided(parser)
    if args.stage == "distill" and "lr" not in args._cli_provided and not args.config:
        args.lr = 1e-4
    if args.diversity not in {"svgd_func", "none"}:
        raise ValueError("Fast trainer supports svgd_func/none; use legacy trainer for DPP/parameter SVGD")
    if args.global_batch < 1 or args.epochs < 1 or args.M < 1 or not 0 <= args.p_drop <= 1:
        raise ValueError("Invalid training dimensions/probability")
    if args.micro_batch_size < 1 or args.max_batch_tokens < 1 or args.loss_chunk < 1:
        raise ValueError("Microbatch and loss chunk limits must be positive")
    if args.stage == "distill" and (not args.teacher_adapter or (not args.weights_dir and not args.uniform_weights)):
        raise ValueError("Distillation needs --teacher_adapter and --weights_dir (or --uniform_weights)")
    return args


class PreparedDataset:
    """Tokenize once per rank; cache in CPU RAM for all epochs."""
    def __init__(self, dataset):
        self.dataset, self.cache = dataset, {}

    def __getitem__(self, index):
        if index not in self.cache:
            example = self.dataset[index]
            if not build_segments(example):
                raise ValueError(f"No supervised response tokens: {example.example_id}")
            # TrainExample.input_ids constructs a tensor on every property
            # access. Materialize once and drop redundant Python token/label
            # lists; the frozen labels are represented by spans below.
            self.cache[index] = SimpleNamespace(
                example_id=example.example_id, input_ids=example.input_ids,
                step_spans=example.step_spans, answer_span=example.answer_span)
        return self.cache[index]


def collate(examples, pad_token_id, device):
    length = max(example.input_ids.numel() for example in examples)
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, example in enumerate(examples):
        n = example.input_ids.numel()
        ids[b, :n], mask[b, :n] = example.input_ids, 1
    ids, mask = ids.to(device), mask.to(device)
    labels = torch.roll(ids, -1, dims=1)
    return ids, mask, labels


def forward_logits(model, ids, mask, context):
    # PEFT keeps fp32 master adapters. Autocast avoids running their large input
    # projections in fp32 while retaining fp32 optimizer/gradient accumulation.
    with torch.autocast(context.device.type, dtype=torch.bfloat16,
                        enabled=context.device.type == "cuda"):
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits


def key_positions(example, index, args):
    spans = build_segments(example)
    key_mask = None
    if args.keystep_dir:
        path = Path(args.keystep_dir) / f"{example.example_id}.npz"
        with np.load(path) as cached:
            key_mask = cached["key_mask"]
        if len(key_mask) != len(spans):
            raise ValueError(f"Key-step cache/tokenizer mismatch: {path}")
    positions = []
    for i, (start, end) in enumerate(spans):
        if key_mask is None or key_mask[i]:
            positions.extend(range(max(start, 1) - 1, min(end, example.input_ids.numel()) - 1))
    positions = torch.tensor(positions, dtype=torch.long)
    if args.dpp_max_tokens and len(positions) > args.dpp_max_tokens:
        generator = torch.Generator().manual_seed(args.seed * 1000003 + index)
        chosen = torch.randperm(len(positions), generator=generator)[:args.dpp_max_tokens].sort().values
        positions = positions[chosen]
    return positions


def phase1_backward(examples, *, indices, model, names, tokenizer, args, context, epoch, denominator):
    ids, mask, labels = collate(examples, tokenizer.pad_token_id, context.device)
    div_on = args.diversity == "svgd_func" and epoch >= args.warmup_div_epochs and len(names) > 1
    positions = [key_positions(example, indices[example.example_id], args).to(context.device)
                 for example in examples] if div_on else []
    detached = [[] for _ in examples]
    if div_on:
        model.eval()
        with torch.no_grad():
            for name in names:
                model.set_adapter(name)
                logits = forward_logits(model, ids, mask, context)
                for b, pos in enumerate(positions):
                    detached[b].append(logits[b].index_select(0, pos).float().log_softmax(-1))
                del logits
    model.train()
    nll_sum = rep_sum = 0.0
    for m, name in enumerate(names):
        model.set_adapter(name)
        token_weights = torch.zeros_like(ids, dtype=torch.float32)
        for b, example in enumerate(examples):
            spans = build_segments(example)
            generator = torch.Generator().manual_seed(args.seed + 1 + epoch * 1000003 + indices[example.example_id])
            keep = torch.bernoulli(torch.full((len(names), len(spans)), 1 - args.p_drop), generator=generator)[m]
            token_weights[b] = step_token_weights(ids.shape[1], spans, keep, device=context.device)
        logits = forward_logits(model, ids, mask, context)
        nll, _ = weighted_nll_kl(logits, labels, token_weights, chunk=args.loss_chunk)
        repel = nll.new_zeros(())
        if div_on:
            for b, pos in enumerate(positions):
                if len(pos) < 2:
                    continue
                predictions = list(detached[b])
                predictions[m] = logits[b].index_select(0, pos).float().log_softmax(-1)
                term, _ = svgd_func_repulsion(predictions, m, args.tr_off_k, args.svgd_sigma or None)
                repel = repel + term
        ((nll + repel) / denominator).backward()
        nll_sum += float(nll.detach())
        rep_sum += float(repel.detach())
        del logits, nll, repel
    return {"nll": nll_sum, "kl": 0.0, "repel": rep_sum, "examples": len(examples)}


def distill_backward(examples, *, model, tokenizer, args, context, denominator):
    ids, mask, labels = collate(examples, tokenizer.pad_token_id, context.device)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    for b, example in enumerate(examples):
        spans = build_segments(example)
        cached = None if args.uniform_weights else load_cached_w(args.weights_dir, example.example_id)
        if cached is None and not args.uniform_weights:
            raise FileNotFoundError(f"Missing council weights: {example.example_id}")
        segment = segment_weights(cached, len(spans), args.uniform_weights)
        if segment is None:
            raise ValueError(f"Council cache/tokenizer mismatch: {example.example_id}")
        weights[b] = step_token_weights(ids.shape[1], spans, segment, device=context.device)
    teacher = None
    if args.lambda_kl:
        model.set_adapter("teacher")
        model.eval()
        with torch.no_grad():
            teacher = forward_logits(model, ids, mask, context).detach()
    model.set_adapter("student")
    set_lora_trainable(model, ["student"], True)
    model.train()
    logits = forward_logits(model, ids, mask, context)
    nll, kl = weighted_nll_kl(logits, labels, weights, teacher=teacher, chunk=args.loss_chunk)
    ((nll + args.lambda_kl * kl) / denominator).backward()
    return {"nll": float(nll.detach()), "kl": float(kl.detach()), "repel": 0.0, "examples": len(examples)}


def checkpoint_signature(args, context):
    from train.pipeline_b200 import digest
    # Changing memory limits is safe; changing the objective/partition is not.
    excluded = {"resume", "resume_from", "dry_run", "log_every", "checkpoint_every",
                "save_every", "save_epochs", "micro_batch_size", "max_batch_tokens", "loss_chunk"}
    signature = {**{k: v for k, v in vars(args).items() if not k.startswith("_") and k not in excluded},
                 "world_size": context.world_size,
                 "train_sha256": digest(args.train_file),
                 "model_config_sha256": digest(Path(args.model_name) / "config.json")}
    for kind in ("keystep_dir", "weights_dir"):
        directory = getattr(args, kind, None)
        if directory:
            # Pipeline markers bind caches to their model/data/recipe. Direct
            # callers still get strict ID/segment checks inside the loss path.
            marker = Path(directory).parent / ("key.done.json" if kind == "keystep_dir" else "weights.done.json")
            if marker.exists():
                signature[kind + "_manifest_sha256"] = digest(marker)
    if args.teacher_adapter:
        teacher = Path(local_adapter_path(args.teacher_adapter))
        signature["teacher_sha256"] = digest(teacher / "adapter_model.safetensors")
    return signature


def save_checkpoint(path, params, optimizer, scheduler, context, signature, epoch, window, step,
                    token_budget, configured_budget):
    states = [None] * context.world_size
    state = rng_state(context.device)
    state.update(token_budget=token_budget, configured_budget=configured_budget)
    if context.world_size > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if context.rank == 0:
        payload = {"signature": signature, "epoch": epoch, "window": window, "step": step,
                   "params": [p.detach().cpu() for p in params], "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "rng": states}
        temporary = str(path) + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, path)
    context.barrier()


def main():
    args = parse_args()
    context = ParallelContext.initialize()
    try:
        run(args, context)
    finally:
        context.close()


def run(args, context):
    from core.proctitle import set_proc_title
    set_proc_title(f"cored-{args.stage}-r{context.rank}")
    set_seed(args.seed)  # identical LoRA initialization on all replicas
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    args.model_name = local_model_path(args.model_name)
    out = Path(args.out_dir)
    if context.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    context.barrier()
    signature = checkpoint_signature(args, context)
    signature_path = out / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError(f"Output configuration differs; choose a new --out_dir: {out}")
    if context.rank == 0:
        write_json(str(signature_path), signature)
    context.barrier()
    if (out / "complete.json").exists():
        if context.rank == 0:
            print(f"[b200] already complete: {out}", flush=True)
        return
    tokenizer = load_tokenizer(args.model_name)
    dtype = torch.float32 if args.cpu_float32 else torch.bfloat16
    base = load_base_model(args.model_name, dtype=dtype, device_map={"": str(context.device)},
                           attn_implementation=args.attn_implementation)
    if args.stage == "phase1":
        config = make_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
        model, names = attach_experts(base, args.M, config)
        set_lora_trainable(model, names, True)
    else:
        from peft import PeftModel
        teacher = local_adapter_path(args.teacher_adapter)
        model = PeftModel.from_pretrained(base, teacher, adapter_name="teacher", local_files_only=True)
        model.load_adapter(teacher, adapter_name="student", is_trainable=True, local_files_only=True)
        set_lora_trainable(model, ["teacher"], False)
        set_lora_trainable(model, ["student"], True)
        names = ["student"]
    model.to(context.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for _, p in lora_parameters(model, names)]
    if context.world_size > 1:
        for parameter in params:
            dist.broadcast(parameter.data, src=0)
    # Dropout streams differ across data shards; all are determined by seed 42.
    set_seed(args.seed + context.rank)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), fused=context.device.type == "cuda")
    source = StepDataset(args.train_file, tokenizer, cutoff_len=args.cutoff_len,
                         segmentation_mode=args.segmentation_mode, limit=args.max_samples)
    if not len(source):
        raise ValueError("Training data is empty")
    ids = [source.example_id(i) for i in range(len(source))]
    if len(set(ids)) != len(ids):
        raise ValueError("Training example IDs must be unique")
    indices = {value: i for i, value in enumerate(ids)}
    dataset = PreparedDataset(source)
    # Shared deterministic lengths allow balancing ranks without activation
    # communication or changing which examples belong to an optimizer update.
    lengths = [dataset[i].input_ids.numel() for i in range(len(source))]
    windows_per_epoch = math.ceil(len(source) / args.global_batch)
    scheduler = cosine_schedule_with_warmup(optimizer, windows_per_epoch * args.epochs, args.warmup_ratio)
    if context.rank == 0:
        save_run_config(str(out), args, {"world_size": context.world_size, "n_examples": len(source),
                                        "total_optimizer_steps": windows_per_epoch * args.epochs,
                                        "mixed_precision": "bf16_autocast" if context.device.type == "cuda" else "cpu_test",
                                        "gradient_sync": "sum_once_per_global_batch"})
    start_epoch = start_window = step = 0
    token_budget = args.max_batch_tokens
    checkpoint = Path(args.resume_from) if args.resume_from else out / "resume.pt"
    if checkpoint.exists() and (args.resume or args.resume_from):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["signature"] != signature or len(saved["params"]) != len(params):
            raise ValueError("Checkpoint configuration/parameter mismatch")
        with torch.no_grad():
            for p, value in zip(params, saved["params"]):
                p.copy_(value.to(p))
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"][context.rank], context.device)
        rank_state = saved["rng"][context.rank]
        if rank_state.get("configured_budget") == args.max_batch_tokens:
            token_budget = rank_state.get("token_budget", args.max_batch_tokens)
        start_epoch, start_window, step = saved["epoch"], saved["window"], saved["step"]
    elif checkpoint.exists():
        raise FileExistsError(f"Checkpoint exists; pass --resume or choose another output directory: {checkpoint}")
    context.barrier()
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
        torch.cuda.reset_peak_memory_stats(context.device)
    started = time.perf_counter()
    all_examples = oom_retries = 0
    stopped = False
    for epoch in range(start_epoch, args.epochs):
        order = list(source.iter_epoch(args.seed, epoch))
        for window, (local, count) in enumerate(global_windows(
                order, args.global_batch, context.rank, context.world_size, lengths)):
            if epoch == start_epoch and window < start_window:
                continue
            optimizer.zero_grad(set_to_none=True)
            stats = {"nll": 0.0, "kl": 0.0, "repel": 0.0, "examples": 0}
            ortho_value = 0.0
            if args.stage == "phase1" and args.lambda_rep and len(names) > 1:
                set_lora_trainable(model, names, True)
                ortho = lora_diversity_loss(model, names, metric="dual_ortho")
                if isinstance(ortho, tuple):
                    ortho = ortho[0]
                (args.lambda_rep * ortho / context.world_size).backward()
                ortho_value = float(ortho.detach())
                del ortho
            examples = [dataset[i] for i in local]
            for batch in pack_microbatches(examples, args.micro_batch_size, token_budget):
                def backward(items):
                    if args.stage == "phase1":
                        return phase1_backward(items, indices=indices, model=model, names=names,
                                               tokenizer=tokenizer, args=args, context=context,
                                               epoch=epoch, denominator=count)
                    return distill_backward(items, model=model, tokenizer=tokenizer, args=args,
                                            context=context, denominator=count)
                values, retries = backward_with_retry(batch, backward, params, context.device)
                oom_retries += retries
                if retries:
                    # Remember a lower budget instead of paying the same OOM on
                    # every later batch. Each rank can pack independently.
                    padded_tokens = len(batch) * max(item.input_ids.numel() for item in batch)
                    token_budget = min(token_budget, max(1, padded_tokens // 2))
                for key, value in values.items():
                    stats[key] += value
            sync_gradients(params, context)
            norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            all_examples += count
            numbers = torch.tensor([stats[key] for key in ("nll", "kl", "repel", "examples")],
                                    dtype=torch.float64, device=context.device)
            context.sum(numbers)
            if step % args.log_every == 0 or step == 1:
                if context.rank == 0:
                    record = {"step": step, "epoch": epoch + 1, "batch_size": count,
                              "nll": numbers[0].item() / count, "kl": numbers[1].item() / count,
                              "repel": numbers[2].item() / count, "ortho": ortho_value,
                              "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0],
                              "elapsed_sec": time.perf_counter() - started,
                              "rank0_token_budget": token_budget}
                    with (out / "train_log.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(f"[b200] {args.stage} {record}", flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(out / "resume.pt", params, optimizer, scheduler, context,
                                signature, epoch, window + 1, step, token_budget, args.max_batch_tokens)
            if args.dry_run and step >= args.dry_run:
                stopped = True
                break
        if stopped:
            save_checkpoint(out / "resume.pt", params, optimizer, scheduler, context,
                            signature, epoch, window + 1, step, token_budget, args.max_batch_tokens)
            break
        if args.save_epochs:
            save_checkpoint(out / "resume.pt", params, optimizer, scheduler, context,
                            signature, epoch + 1, 0, step, token_budget, args.max_batch_tokens)
    context.barrier()
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(context.device) if context.device.type == "cuda" else 0
    telemetry = {"rank": context.rank, "elapsed_sec": elapsed, "peak_allocated_bytes": peak,
                 "oom_retries": oom_retries, "examples_processed_this_invocation": all_examples,
                 "world_size": context.world_size, "optimizer_steps": step, "dry_run": stopped,
                 "global_examples_per_second": all_examples / max(elapsed, 1e-9),
                 "final_token_budget": token_budget,
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(context.device) if peak else 0,
                 "device_name": torch.cuda.get_device_name(context.device) if peak else "cpu"}
    write_json(str(out / f"timing-rank-{context.rank}.json"), telemetry)
    if context.rank == 0:
        model.save_pretrained(str(out), selected_adapters=names)
        tokenizer.save_pretrained(str(out))
        if not stopped:
            write_json(str(out / "complete.json"), {"optimizer_steps": step, "signature": signature})
    context.barrier()


if __name__ == "__main__":
    main()
