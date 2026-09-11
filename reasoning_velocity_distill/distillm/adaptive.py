"""Micro-batch routing with evaluation-only, monotonic ON-policy milestones."""

from dataclasses import dataclass, fields
import math
import random

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class AdaptiveConfig:
    rho_min: float = 0.1
    rho_max: float = 0.5
    rho_increment: float = 0.1
    progress_signal: str = "loss"
    loss_improvement_threshold: float = 0.05
    metric_improvement_threshold: float = 1.0
    eval_ema_beta: float = 0.9
    min_evals_between_rho_updates: int = 1
    off_ema_beta: float = 0.9
    off_min_absorption: float = 0.2
    off_plateau_threshold: float = 0.01
    off_transition_patience: int = 3
    eps: float = 1e-6

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, f.name, f.default) for f in fields(cls)})

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name != "progress_signal" and not math.isfinite(value):
                raise ValueError(f"--{f.name.replace('_', '-')} must be finite")
        if not 0 <= self.rho_min <= self.rho_max <= 1:
            raise ValueError("Require 0 <= --rho-min <= --rho-max <= 1")
        if self.rho_increment <= 0 or self.eps <= 0:
            raise ValueError("--rho-increment and --eps must be positive")
        if self.progress_signal not in ("loss", "metric", "either"):
            raise ValueError("--progress-signal must be loss, metric, or either")
        if not 0 <= self.eval_ema_beta < 1 or not 0 <= self.off_ema_beta < 1:
            raise ValueError("EMA betas must be in [0, 1)")
        if self.loss_improvement_threshold <= 0 or self.metric_improvement_threshold <= 0:
            raise ValueError("Progress improvement thresholds must be positive")
        if not 0 <= self.off_min_absorption <= 1 or self.off_plateau_threshold < 0:
            raise ValueError("OFF absorption must be in [0, 1]; plateau threshold must be nonnegative")
        for name in ("min_evals_between_rho_updates", "off_transition_patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")


class AdaptiveScheduler:
    MODES = ("off_policy", "privileged", "on_policy")

    def __init__(self, config, seed=42):
        self.config = config
        self.rng = random.Random(seed)
        self.base_mode = "off"
        self.rho_on = config.rho_min
        self.off_initial_loss = self.off_loss_ema = self.previous_eval_ema_off = None
        self.eval_loss_ema = self.eval_loss_reference = self.metric_reference = None
        self.off_transition_counter = 0
        self.eval_count = 0
        self.last_rho_update_eval = 0
        self.counts = dict.fromkeys(self.MODES, 0)

    def route(self, device=None):
        """Call exactly once per training DataLoader batch on every rank."""
        distributed = dist.is_available() and dist.is_initialized()
        mode_id = 0
        if not distributed or dist.get_rank() == 0:
            mode_id = (2 if self.rng.random() < self.rho_on
                       else int(self.base_mode == "privileged"))
        if distributed:
            # NCCL requires CUDA tensors; Gloo uses CPU even during GPU tests.
            routing_device = (device if device is not None else torch.cuda.current_device()) \
                if dist.get_backend() == "nccl" else "cpu"
            decision = torch.tensor(mode_id, dtype=torch.long, device=routing_device)
            dist.broadcast(decision, src=0)
            mode_id = int(decision.item())
        mode = self.MODES[mode_id]
        self.counts[mode] += 1
        return mode

    def observe_off_loss(self, loss):
        """Observe the existing OFF distillation objective, averaged across ranks."""
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite OFF-policy loss")
        if self.base_mode != "off":
            return
        if self.off_loss_ema is None:
            self.off_initial_loss = self.off_loss_ema = loss
        else:
            beta = self.config.off_ema_beta
            self.off_loss_ema = beta * self.off_loss_ema + (1 - beta) * loss

    def on_evaluation(self, eval_loss, metric=None):
        """Update once per dev evaluation, after a completed optimizer step.

        The first event establishes references. Missing OFF observations cannot
        count as plateau evidence. Returned counters cover the interval that just
        ended, while base_mode/rho_on describe routing for the next interval.
        """
        if not math.isfinite(eval_loss) or (metric is not None and not math.isfinite(metric)):
            raise FloatingPointError("Non-finite scheduler evaluation signal")
        c = self.config
        if c.progress_signal == "metric" and metric is None:
            raise ValueError("Metric progress requires an evaluation metric")
        self.eval_count += 1
        self.eval_loss_ema = (eval_loss if self.eval_loss_ema is None else
                              c.eval_ema_beta * self.eval_loss_ema + (1 - c.eval_ema_beta) * eval_loss)
        if self.eval_loss_reference is None:
            self.eval_loss_reference = self.eval_loss_ema
            self.last_rho_update_eval = self.eval_count
        if metric is not None and self.metric_reference is None:
            self.metric_reference = metric

        loss_improvement = (self.eval_loss_reference - self.eval_loss_ema) / (self.eval_loss_reference + c.eps)
        metric_improvement = None if metric is None else metric - self.metric_reference
        loss_ready = c.progress_signal in ("loss", "either") and loss_improvement >= c.loss_improvement_threshold
        metric_ready = (c.progress_signal in ("metric", "either") and metric_improvement is not None
                        and metric_improvement >= c.metric_improvement_threshold)
        rho_updated = False
        if ((loss_ready or metric_ready) and self.rho_on < c.rho_max
                and self.eval_count - self.last_rho_update_eval >= c.min_evals_between_rho_updates):
            self.rho_on = min(self.rho_on + c.rho_increment, c.rho_max)
            # Both references correspond to the same last-increase milestone.
            self.eval_loss_reference = self.eval_loss_ema
            self.metric_reference = metric
            self.last_rho_update_eval = self.eval_count
            rho_updated = True

        absorption = progress = None
        transitioned = False
        if self.base_mode == "off":
            if self.counts["off_policy"] and self.off_loss_ema is not None:
                absorption = (self.off_initial_loss - self.off_loss_ema) / (self.off_initial_loss + c.eps)
                if self.previous_eval_ema_off is not None:
                    progress = (self.previous_eval_ema_off - self.off_loss_ema) / (self.previous_eval_ema_off + c.eps)
                ready = (progress is not None and absorption >= c.off_min_absorption
                         and 0 <= progress <= c.off_plateau_threshold)
                self.off_transition_counter = self.off_transition_counter + 1 if ready else 0
                self.previous_eval_ema_off = self.off_loss_ema
                if self.off_transition_counter >= c.off_transition_patience:
                    self.base_mode = "privileged"
                    transitioned = True
            else:
                self.off_transition_counter = 0

        total = sum(self.counts.values())
        values = dict(
            base_mode=self.base_mode, rho_on=self.rho_on,
            eval_loss=eval_loss, eval_loss_ema=self.eval_loss_ema,
            eval_loss_reference=self.eval_loss_reference, loss_improvement=loss_improvement,
            current_metric=metric, metric_reference=self.metric_reference,
            metric_improvement=metric_improvement, off_loss_ema=self.off_loss_ema,
            off_absorption=absorption, off_progress=progress,
            off_transition_counter=self.off_transition_counter,
            off_batches=self.counts["off_policy"], privileged_batches=self.counts["privileged"],
            on_batches=self.counts["on_policy"], actual_on_ratio=self.counts["on_policy"] / total if total else 0.,
            rho_updated=rho_updated, base_transitioned=transitioned,
        )
        self.counts = dict.fromkeys(self.MODES, 0)
        return {f"scheduler/{key}": value for key, value in values.items()}
