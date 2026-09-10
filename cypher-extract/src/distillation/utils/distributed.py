from __future__ import annotations

import os

import torch
import torch.distributed as dist


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_is_initialized() else int(os.getenv("RANK", "0"))


def get_world_size() -> int:
    return dist.get_world_size() if distributed_is_initialized() else int(os.getenv("WORLD_SIZE", "1"))


def all_gather_tensor(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    group=None,
    operation: str = "cat",
) -> torch.Tensor:
    """Gather equal-shaped tensors and concatenate or stack them.

    A non-distributed process returns its input for `cat`, and an added world
    dimension for `stack`. This makes evaluation helpers usable in CPU tests.
    """

    if operation not in {"cat", "stack"}:
        raise ValueError("operation must be 'cat' or 'stack'.")
    if not distributed_is_initialized():
        return tensor if operation == "cat" else tensor.unsqueeze(dim)

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor, group=group)
    if operation == "cat":
        return torch.cat(gathered, dim=dim)
    return torch.stack(gathered, dim=dim)
