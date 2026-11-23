"""CUDA timing and memory profiling for comparable distillation benchmarks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback


_TSV_HEADER = (
    "method\tworld_size\tper_device_batch_size\tgradient_accumulation_steps\t"
    "global_microbatch_size\tmeasured_optimizer_steps\tavg_optimizer_step_ms\t"
    "avg_batch_ms\tpeak_allocated_gib\tpeak_reserved_gib\n"
)


class CudaBenchmarkCallback(TrainerCallback):
    """Profile full optimizer steps after an initial warm-up period.

    ``on_step_begin``/``on_step_end`` enclose all accumulated microbatches plus
    backward, optimizer, and scheduler work for one optimizer update. Every
    process synchronizes its CUDA stream, and the reported duration/memory is
    the maximum across DDP ranks because the slowest rank determines throughput.
    ``avg_batch_ms`` divides the optimizer-step time by gradient accumulation so
    it denotes one global microbatch (one per-device batch on every GPU).
    """

    def __init__(self) -> None:
        self._step_start: Optional[float] = None
        self._step_microbatches = 0
        self._optimizer_step_ms: list[float] = []
        self._measured_microbatches = 0
        self._peak_allocated_bytes = 0.0
        self._peak_reserved_bytes = 0.0

    @staticmethod
    def _cuda_ready() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def _distributed_max(values: list[float]) -> list[float]:
        tensor = torch.tensor(values, dtype=torch.float64, device=torch.cuda.current_device())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.cpu().tolist()

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._cuda_ready():
            raise RuntimeError("--benchmark_enabled requires CUDA.")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._step_start = time.perf_counter()
        self._step_microbatches = 1

    def on_substep_end(self, args, state, control, **kwargs):
        # Fired after each non-sync gradient-accumulation microbatch. The final
        # sync microbatch ends through on_step_end and is represented by the
        # initial count of one set in on_step_begin.
        self._step_microbatches += 1

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is None:
            return

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self._step_start) * 1000.0
        peak_allocated = float(torch.cuda.max_memory_allocated())
        peak_reserved = float(torch.cuda.max_memory_reserved())
        elapsed_ms, peak_allocated, peak_reserved = self._distributed_max(
            [elapsed_ms, peak_allocated, peak_reserved]
        )

        warmup_steps = max(int(getattr(args, "benchmark_warmup_steps", 0)), 0)
        if int(state.global_step) > warmup_steps:
            self._optimizer_step_ms.append(elapsed_ms)
            self._measured_microbatches += self._step_microbatches
            self._peak_allocated_bytes = max(self._peak_allocated_bytes, peak_allocated)
            self._peak_reserved_bytes = max(self._peak_reserved_bytes, peak_reserved)
        self._step_start = None

    def on_train_end(self, args, state, control, **kwargs):
        # The DDP max-reduction above gives every rank the same measurements;
        # only world-process zero writes the shared summary file.
        if not bool(getattr(state, "is_world_process_zero", True)):
            return

        measured_steps = len(self._optimizer_step_ms)
        if measured_steps == 0:
            raise RuntimeError(
                "Benchmark collected no measured steps. Reduce --benchmark_warmup_steps "
                "or increase the amount of training data."
            )

        grad_accum = max(int(args.gradient_accumulation_steps), 1)
        world_size = int(getattr(args, "world_size", 1))
        per_device_batch = int(args.per_device_train_batch_size)
        avg_step_ms = sum(self._optimizer_step_ms) / measured_steps
        # Weight by the number of microbatches so a final partial accumulation
        # group does not bias the per-batch number.
        avg_batch_ms = sum(self._optimizer_step_ms) / max(self._measured_microbatches, 1)
        gib = float(1024**3)
        method = getattr(args, "benchmark_method", "") or getattr(args, "kd_loss_type", "unknown")

        output_file = getattr(args, "benchmark_output_file", None)
        if not output_file:
            output_file = os.path.join(args.output_dir, "benchmark_results.tsv")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        needs_header = not output_path.exists() or output_path.stat().st_size == 0
        with output_path.open("a", encoding="utf-8") as handle:
            if needs_header:
                handle.write(_TSV_HEADER)
            handle.write(
                f"{method}\t{world_size}\t{per_device_batch}\t{grad_accum}\t"
                f"{world_size * per_device_batch}\t{measured_steps}\t"
                f"{avg_step_ms:.3f}\t{avg_batch_ms:.3f}\t"
                f"{self._peak_allocated_bytes / gib:.3f}\t"
                f"{self._peak_reserved_bytes / gib:.3f}\n"
            )

        print(
            "BENCHMARK_RESULT "
            f"method={method} measured_steps={measured_steps} "
            f"avg_batch_ms={avg_batch_ms:.3f} "
            f"peak_allocated_gib={self._peak_allocated_bytes / gib:.3f} "
            f"peak_reserved_gib={self._peak_reserved_bytes / gib:.3f}"
        )
