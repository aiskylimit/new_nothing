"""EMA teacher wrapper for the TROPIC-G-EMA exploratory ablation.

NOT part of the verified real OPSD/TROPIC-G recipe: OPSD's own real 4B
launch script (scripts/run_opsd_4b.sh, github.com/siyan-zhao/OPSD, checked
live on 2026-09-16) passes `--fixed_teacher`, NOT `--use_ema_teacher` - the
EMA option exists in their code (opsd_train.py's `use_ema_teacher`/
`ema_decay` fields, opsd_trainer.py's `EMAUpdateCallback`/`_update_ema`/
`_ema_teacher_context`) but is OPTIONAL and unused in their reported 4B
numbers (`use_ema_teacher=True` and `fixed_teacher=True` are documented as
mutually exclusive there). This file reuses their exact EMA formula
(`ema = decay * ema + (1 - decay) * student`, their opsd_trainer.py lines
529/554) and lazy-init behavior (first call snapshots current weights as the
initial EMA state) adapted to this project's own pattern for teacher
strategies that `tropic.model.ContextualPolicy` doesn't natively express
(always-live or permanently-frozen-at-init only) - same reasoning, and same
swap-in/restore-live-weights mechanics, as `tropic.rlsd.PeriodicTeacherSync`
(that class does a periodic HARD sync every N steps instead of a per-step
exponential blend; this one blends every step).

Only trainable (LoRA) tensors are snapshotted/blended - the frozen base
model never changes, so a full second model copy is never needed, same
convention as PeriodicTeacherSync.
"""
from __future__ import annotations

import contextlib

import torch
from torch import Tensor


class EMATeacherSync:
    """Teacher = exponential moving average of the student's own (LoRA)
    trainable weights, updated once per completed training step (call
    `update()` AFTER `optimizer.step()`). Exposes wrapped versions of
    `tropic.model.ContextualPolicy`'s `forward_teacher_logits` and
    `forward_teacher_context_only` (the two TROPIC-G/Eq.11 needs) - each
    temporarily swaps the EMA snapshot onto the model's trainable tensors,
    calls through to the real policy method, then restores the live
    (training) values, so a teacher forward pass never perturbs the weights
    being trained."""

    def __init__(self, model, decay: float):
        self.model = model
        self.decay = decay
        self._ema_params: dict[str, Tensor] | None = None

    def _trainable_named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def update(self) -> None:
        """Call once per completed training step, AFTER optimizer.step().
        Lazily initializes the EMA state as an exact copy of the CURRENT
        (post-step) weights on the very first call - matching OPSD's own
        `_update_ema`'s lazy-init behavior - so the teacher's first-ever
        forward pass (step 0, before any `update()` call) uses the model's
        live init weights (see `_ema_weights` below), identical to what
        `fixed_teacher=True` would have scored at step 0."""
        if self._ema_params is None:
            self._ema_params = {n: p.detach().clone() for n, p in self._trainable_named_parameters()}
            return
        with torch.no_grad():
            for name, param in self._trainable_named_parameters():
                ema = self._ema_params[name]
                ema.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def _ema_weights(self):
        if self._ema_params is None:
            yield  # not yet initialized (before the first update()) - use live weights
            return
        params = self._trainable_named_parameters()
        live = {n: p.detach().clone() for n, p in params}
        with torch.no_grad():
            for n, p in params:
                p.data.copy_(self._ema_params[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in params:
                    p.data.copy_(live[n])

    def forward_teacher_logits(self, policy, teacher_prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        with self._ema_weights():
            return policy.forward_teacher_logits(teacher_prefix_ids, generated_ids)

    def forward_teacher_context_only(self, policy, context_only_prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        with self._ema_weights():
            return policy.forward_teacher_context_only(context_only_prefix_ids, generated_ids)
