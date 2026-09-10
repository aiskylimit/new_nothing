#!/usr/bin/env python3
"""Fail-fast validation for the isolated multi-GPU SDXL environment."""

from __future__ import annotations

import json
import os

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    expected = int(os.environ.get("NUM_GPUS", "8"))
    if torch.cuda.device_count() != expected:
        raise RuntimeError(
            f"Expected {expected} visible GPUs, got {torch.cuda.device_count()}. "
            "Check GPU_IDS/NUM_GPUS/CUDA_VISIBLE_DEVICES."
        )
    family = os.environ.get("TARGET_GPU_FAMILY", "B200").upper()
    cuda_runtime = torch.version.cuda or ""
    cuda_parts = tuple(int(part) for part in cuda_runtime.split(".")[:2] if part.isdigit())
    if family == "B200" and cuda_parts and cuda_parts < (12, 8):
        raise RuntimeError(f"CUDA runtime {cuda_runtime} is too old for the B200 environment")
    devices = []
    for index in range(expected):
        device = torch.device(f"cuda:{index}")
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        if family == "B200" and capability[0] < 10:
            raise RuntimeError(f"GPU {index} {name!r} is not Blackwell: capability={capability}")
        if family == "A100" and "A100" not in name.upper():
            raise RuntimeError(f"GPU {index} {name!r} does not match TARGET_GPU_FAMILY=A100")
        left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16, requires_grad=True)
        right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
        loss = (left @ right).float().square().mean()
        loss.backward()
        if left.grad is None or not torch.isfinite(left.grad).all():
            raise RuntimeError(f"GPU {index} BF16 forward/backward produced invalid gradients")
        devices.append({
            "index": index,
            "name": name,
            "capability": list(capability),
            "memory_gib": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2),
        })

    import accelerate
    import diffusers
    import transformers
    from diffusers import StableDiffusionXLPipeline  # noqa: F401

    payload = {
        "target_gpu_family": family,
        "devices": devices,
        "torch": torch.__version__,
        "cuda_runtime": cuda_runtime,
        "accelerate": accelerate.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "bf16_forward_backward": "passed",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
