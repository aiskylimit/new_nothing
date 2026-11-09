"""Data-parallel building blocks for CORED's multiple-adapter backward passes."""
from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class ParallelContext:
    rank: int
    world_size: int
    device: torch.device

    @classmethod
    def initialize(cls):
        world = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if torch.cuda.is_available():
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        if world > 1:
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=30))
        return cls(rank, world, device)

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def sum(self, value):
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def close(self):
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def sync_gradients(params, context, bucket_bytes=64 * 1024 * 1024):
    """Sum gradients once per optimizer update. Loss already divides by global B.

    Explicit synchronization avoids DDP reducer pitfalls with set_adapter and
    separate backward calls. Every rank participates, including an empty shard
    of a partial final batch. No forward/activation communication is needed.
    """
    if context.world_size == 1:
        return
    buckets, bucket, size = [], [], 0
    for parameter in params:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        count = parameter.numel() * parameter.element_size()
        if bucket and (size + count > bucket_bytes or parameter.dtype != bucket[0].dtype):
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append(parameter)
        size += count
    if bucket:
        buckets.append(bucket)
    for bucket in buckets:
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        context.sum(flat)
        offset = 0
        for parameter in bucket:
            parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()


def rng_state(device):
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def backward_with_retry(items, backward, params, device):
    """Retry OOMs with smaller microbatches, without double-counting gradients.

    A later expert may OOM after earlier experts already backpropagated. Snapshot
    LoRA gradients and RNG so retries roll back the *whole* microbatch.
    """
    before = [None if p.grad is None else p.grad.detach().clone() for p in params]
    state = rng_state(device)
    failed = False
    try:
        result = backward(items)
    except torch.OutOfMemoryError:
        for parameter, saved in zip(params, before):
            parameter.grad = saved
        restore_rng(state, device)
        failed = True
    if not failed:
        return result, 0
    del before
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(items) <= 1:
        raise RuntimeError("One sequence exceeds memory; reduce --cutoff_len or enable checkpointing")
    middle = len(items) // 2
    left, nl = backward_with_retry(items[:middle], backward, params, device)
    right, nr = backward_with_retry(items[middle:], backward, params, device)
    return {key: left[key] + right[key] for key in left}, nl + nr + 1


def global_windows(order, global_batch, rank, world_size, lengths=None):
    if global_batch < 1:
        raise ValueError("global_batch must be positive")
    for start in range(0, len(order), global_batch):
        ids = order[start:start + global_batch]
        if lengths is None:
            yield ids[rank::world_size], len(ids)
            continue
        # Preserve the shuffled global batch, balance long-sequence attention
        # work within it, and cap each rank at ceil(B/world) examples.
        shards, costs = [[] for _ in range(world_size)], [0] * world_size
        capacity = (len(ids) + world_size - 1) // world_size
        for index in sorted(ids, key=lambda i: lengths[i], reverse=True):
            target = min((r for r in range(world_size) if len(shards[r]) < capacity),
                         key=lambda r: (costs[r], len(shards[r]), r))
            shards[target].append(index)
            costs[target] += lengths[index] ** 2
        yield shards[rank], len(ids)


def pack_microbatches(items, max_sequences, max_tokens):
    """Bucket similar lengths inside an optimizer window; never drop or repeat."""
    if max_sequences < 1 or max_tokens < 1:
        raise ValueError("Microbatch limits must be positive")
    batch, longest = [], 0
    for item in sorted(items, key=lambda x: x.input_ids.numel(), reverse=True):
        length = item.input_ids.numel()
        if batch and (len(batch) >= max_sequences or max(longest, length) * (len(batch) + 1) > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(item)
        longest = max(longest, length)
    if batch:
        yield batch
