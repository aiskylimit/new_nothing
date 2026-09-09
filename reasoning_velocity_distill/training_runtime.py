import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from utils import print_rank, save_rank, get_rank


def distributed_context():
    return (dist.get_world_size(), get_rank()) if dist.is_initialized() else (1, 0)


def log_message(args, message):
    print_rank(message, flush=True)
    if not dist.is_initialized() or get_rank() == 0:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    save_rank(message, Path(args.save) / "log.txt")


def make_loader(args, dataset, training=False):
    world_size, rank = distributed_context()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=training, seed=args.seed, drop_last=training)
    return DataLoader(dataset, sampler=sampler,
                      batch_size=args.batch_size if training else args.eval_batch_size,
                      num_workers=args.num_workers, collate_fn=dataset.collate,
                      drop_last=training)


def configure_training_steps(args, micro_batches):
    """Use complete accumulation windows so epoch/checkpoint counts stay aligned."""
    args.train_iters_per_epoch = micro_batches // args.gradient_accumulation_steps
    if args.train_iters_per_epoch < 1:
        raise ValueError("Training data must fill one optimizer step; reduce batch size, "
                         "GPU count or gradient accumulation steps")
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.epochs is None:
        args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
    if args.total_iters > args.epochs * args.train_iters_per_epoch:
        raise ValueError("--epochs cannot supply the requested --total-iters")
    for field in ("eval_interval", "save_interval"):
        if getattr(args, field) == -1:
            setattr(args, field, args.train_iters_per_epoch)


def setup_model_and_optimizer(args, ds_config, device):
    # Runtime dependencies are loaded here so the loss/data helpers also work on CPU.
    import deepspeed
    from utils import get_model, get_optimizer_params, get_optimizer_params_peft
    from transformers import (get_constant_schedule_with_warmup,
                              get_cosine_schedule_with_warmup,
                              get_polynomial_decay_schedule_with_warmup)

    model = get_model(args, device)
    optimizer, scheduler = None, None
    if args.do_train:
        groups = get_optimizer_params_peft(args, model) if args.peft else get_optimizer_params(args, model)
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
        print_rank(f"Optimizer = {optimizer.__class__.__name__}")
        if args.lr_decay_style == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_iters)
        elif args.lr_decay_style == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.total_iters, eta_min=args.lr_min)
        elif args.lr_decay_style == "noam":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, args.warmup_iters, args.total_iters, power=0.5)
        elif args.lr_decay_style == "wrmup_cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, args.warmup_ratio * args.total_iters, args.total_iters)
        else:
            raise ValueError(f"Unsupported scheduler: {args.lr_decay_style}")
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, args=args, lr_scheduler=scheduler, config_params=ds_config)
    return engine, optimizer, scheduler


def save_checkpoint(args, tokenizer, model, step):
    destination = Path(args.save) / str(step)
    if not dist.is_initialized() or get_rank() == 0:
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(destination)
        model.module.save_pretrained(destination, safe_serialization=False)
        log_str = f"Saved model to {destination}"
        print_rank(log_str, flush=True)
        save_rank(log_str, Path(args.save) / "log.txt")
    if dist.is_initialized():
        dist.barrier()
