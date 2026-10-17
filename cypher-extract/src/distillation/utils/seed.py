from __future__ import annotations

import random

import numpy as np
import torch

from .distributed import get_rank


def seed_everything(seed: int, *, rank_offset: bool = True, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch and return the effective seed."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    effective_seed = seed + get_rank() if rank_offset else seed
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return effective_seed


def capture_rng_state() -> dict:
    """Capture every RNG stream used by training and stochastic inference."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing: {', '.join(sorted(missing))}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise ValueError("RNG state has no CUDA streams for the current CUDA inference process.")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                f"RNG state contains {len(cuda_state)} CUDA streams, but the current process sees "
                f"{torch.cuda.device_count()} devices."
            )
        torch.cuda.set_rng_state_all(cuda_state)
