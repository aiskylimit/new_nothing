"""Checkpoint milestones and distributed experiment accounting."""

import json
import math
import os
import time

import torch
import torch.distributed as dist


def checkpoint_milestones(total_steps, fractions):
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    result = []
    for fraction in fractions or ():
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Checkpoint fractions must be finite and in (0, 1]")
        result.append((fraction, math.ceil(total_steps * fraction)))
    return result


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def append_json(path, data):
    with open(path, "a") as handle:
        handle.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")


class ExperimentTracker:
    MODES = ("off_policy", "self_distill", "on_policy", "opsd")
    TOKEN_FIELDS = ("student_input", "reference_input", "response",
                    "generated_response", "self_context")

    def __init__(self, save_dir, device, seed, scheduler_type):
        self.save_dir = save_dir
        self.seed = seed
        self.scheduler_type = scheduler_type
        self.started = time.perf_counter()
        self.seconds = {"optimization": 0.0, "dev_evaluation": 0.0, "checkpoint": 0.0}
        self.tokens = torch.zeros((len(self.MODES), len(self.TOKEN_FIELDS)),
                                  dtype=torch.float64, device=device)
        self.counts = dict.fromkeys(self.MODES, 0)
        self.probability_sums = dict.fromkeys(self.MODES, 0.0)

    def add_microbatch(self, mode, student_batch, reference_batch, labels, context):
        zero = labels.new_zeros(())
        response = (labels != -100).sum()
        values = torch.stack((
            student_batch["attention_mask"].sum(),
            reference_batch["attention_mask"].sum() if reference_batch is not None else zero,
            response,
            response if mode in ("on_policy", "opsd") else zero,
            context.sum() if context is not None else zero,
        )).detach().double()
        self.tokens[self.MODES.index(mode)] += values

    def record_step(self, mode, probabilities):
        self.counts[mode] += 1
        for name, value in probabilities.items():
            self.probability_sums[name] += value

    def snapshot(self, step, probabilities, final=False):
        tokens = self.tokens.clone()
        if dist.is_initialized():
            dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        result = {
            "global_step": step, "seed": self.seed, "scheduler": self.scheduler_type,
            "next_step_probabilities": probabilities,
            "mode_steps": self.counts.copy(),
            "mean_probabilities": {m: v / step for m, v in self.probability_sums.items()},
            "realized_mode_ratios": {m: v / step for m, v in self.counts.items()},
            "tokens_by_mode": {m: dict(zip(self.TOKEN_FIELDS, row))
                               for m, row in zip(self.MODES, tokens.tolist())},
            "total_tokens": dict(zip(self.TOKEN_FIELDS, tokens.sum(0).tolist())),
            "seconds": {**self.seconds, "training_loop_wall": time.perf_counter() - self.started},
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            append_json(os.path.join(self.save_dir, "exposure.jsonl"), result)
            if final:
                write_json(os.path.join(self.save_dir, "training_summary.json"), result)
        return result
