"""Check the provisioned CUDA runtime in a short-lived process before training."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_devices", type=int, default=3)
    parser.add_argument("--required_gpu_name")
    args = parser.parse_args()
    if args.expected_devices < 1 or not torch.cuda.is_available() or torch.cuda.device_count() != args.expected_devices:
        raise RuntimeError(f"Expected exactly {args.expected_devices} visible CUDA devices")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("This PyTorch build has no NCCL backend")
    devices = []
    for rank in range(args.expected_devices):
        torch.cuda.set_device(rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device {rank} does not support bf16")
        properties = torch.cuda.get_device_properties(rank)
        if args.required_gpu_name and args.required_gpu_name not in properties.name:
            raise RuntimeError(f"Expected {args.required_gpu_name}, found {properties.name}")
        devices.append({"index": rank, "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "capability": list(torch.cuda.get_device_capability(rank))})
    result = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "nccl": torch.cuda.nccl.version(), "devices": devices,
              "cuda_arch_list": torch.cuda.get_arch_list()}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
