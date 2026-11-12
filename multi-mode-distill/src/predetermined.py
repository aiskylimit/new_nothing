"""A step-only linear exposure schedule, independent of validation losses."""

from dataclasses import dataclass, fields
import math

from .adaptive import AdaptiveScheduler


@dataclass(frozen=True)
class PredeterminedConfig:
    off_init: float = 0.85
    self_init: float = 0.10
    on_init: float = 0.05
    self_increment: float = 0.0001
    on_increment: float = 0.0001
    self_max: float = 0.25
    on_max: float = 0.25
    update_interval: int = 1

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, "predetermined_" + f.name, f.default)
                      for f in fields(cls)})

    def __post_init__(self):
        ratios = (self.off_init, self.self_init, self.on_init)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in ratios):
            raise ValueError("Predetermined initial probabilities must be finite and in [0, 1]")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Predetermined OFF + SELF + ON initial probabilities must sum to 1")
        if (any(not math.isfinite(v) or not 0 <= v <= 1
                for v in (self.self_max, self.on_max))
                or self.self_init > self.self_max or self.on_init > self.on_max
                or self.self_max + self.on_max > 1):
            raise ValueError("Predetermined SELF/ON initial probabilities must fit their maxima; maxima must sum to at most 1")
        if any(not math.isfinite(v) or v < 0
               for v in (self.self_increment, self.on_increment)):
            raise ValueError("Predetermined increments must be finite and nonnegative")
        if not isinstance(self.update_interval, int) or self.update_interval < 1:
            raise ValueError("Predetermined update interval must be a positive integer")


class PredeterminedScheduler(AdaptiveScheduler):
    """Use initial ratios for step 1; update after every N completed updates.

    SELF = min(self_init + n * self_increment, self_max);
    ON = min(on_init + n * on_increment, on_max); OFF = 1 - SELF - ON.
    Each growing mode stops independently at its own maximum.
    Sampling/broadcasting and optimizer-step counting match adaptive routing.
    """

    def __init__(self, config=None, seed=42):
        super().__init__(seed=seed)
        self.config = config or PredeterminedConfig()

    @property
    def mode_probabilities(self):
        c = self.config
        updates = sum(self.counts.values()) // c.update_interval
        self_ratio = min(c.self_init + updates * c.self_increment, c.self_max)
        on_ratio = min(c.on_init + updates * c.on_increment, c.on_max)
        return {
            "off_policy": max(0.0, 1.0 - self_ratio - on_ratio),
            "self_distill": self_ratio,
            "on_policy": on_ratio,
        }

    def on_evaluation(self, *args, **kwargs):
        raise RuntimeError("Predetermined scheduling must not depend on dev losses")
