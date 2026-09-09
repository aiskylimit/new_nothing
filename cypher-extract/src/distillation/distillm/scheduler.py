from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class RolloutSource(StrEnum):
    DATASET = "dataset"
    FRESH = "fresh"
    REPLAY = "replay"


@dataclass(slots=True)
class AdaptiveRolloutScheduler:
    """DistiLLM adaptive scheduler ported from the template training loop."""

    threshold: float = 0.0
    loss_eps: float = 0.1
    replay_ratio: str = "decreasing"
    seed: int = 0
    # CypherKD starts the adaptive comparison baseline at zero rather than
    # evaluating the untrained student before the first training epoch.
    previous_validation_loss: float = 0.0
    _rng: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if self.loss_eps < 0.0:
            raise ValueError("loss_eps must be non-negative.")
        if self.replay_ratio not in {"constant", "increasing", "decreasing"}:
            raise ValueError("Unknown replay_ratio schedule.")
        self._rng = np.random.RandomState(self.seed)

    def fresh_probability(self, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        if self.replay_ratio == "constant":
            return self.threshold * 0.5
        if self.replay_ratio == "increasing":
            return self.threshold * progress
        return self.threshold * (1.0 - progress)

    def choose(self, *, progress: float, replay_size: int, capacity: int, batch_size: int) -> RolloutSource:
        if replay_size < 0 or capacity <= 0 or batch_size <= 0:
            raise ValueError("Invalid replay scheduler sizes.")
        draw = float(self._rng.uniform(0.0, 1.0))
        fresh_probability = self.fresh_probability(progress)

        # This reproduces the template: while the buffer is below capacity,
        # every selected on-policy step creates fresh student generations.
        if draw < fresh_probability or (draw < self.threshold and replay_size < capacity):
            return RolloutSource.FRESH
        if draw < self.threshold and replay_size >= batch_size:
            return RolloutSource.REPLAY
        return RolloutSource.DATASET

    def update(self, validation_loss: float) -> bool:
        """Increase the threshold when loss exceeds the reference comparison baseline."""

        changed = validation_loss >= self.previous_validation_loss + self.loss_eps
        if changed:
            self.threshold = min(1.0, self.threshold + 0.1)
            # The template only advances the comparison baseline after a
            # deterioration event; improvements leave the baseline unchanged.
            self.previous_validation_loss = validation_loss
        return changed

    def state_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "loss_eps": self.loss_eps,
            "replay_ratio": self.replay_ratio,
            "previous_validation_loss": self.previous_validation_loss,
            "random_state": self._rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("loss_eps", self.loss_eps) != self.loss_eps:
            raise ValueError("Scheduler loss_eps does not match the current configuration.")
        if state.get("replay_ratio", self.replay_ratio) != self.replay_ratio:
            raise ValueError("Scheduler replay_ratio does not match the current configuration.")
        self.threshold = float(state["threshold"])
        previous_validation_loss = state.get("previous_validation_loss", 0.0)
        # Old checkpoints created before reference-parity initialization may
        # contain None if they were saved before their initial-eval callback.
        self.previous_validation_loss = 0.0 if previous_validation_loss is None else float(previous_validation_loss)
        if "random_state" in state:
            self._rng.set_state(state["random_state"])
