"""A step-only linear exposure schedule, independent of validation losses."""

from dataclasses import dataclass, fields
import math

from .adaptive import AdaptiveScheduler


@dataclass(frozen=True)
class PredeterminedConfig:
    off_init: float = 0.85
    self_init: float = 0.10
    on_init: float = 0.05
    off_decrement: float = 0.0001
    on_increment: float = 0.0001
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
        if any(not math.isfinite(v) or v < 0
               for v in (self.off_decrement, self.on_increment)):
            raise ValueError("Predetermined increments/decrements must be finite and nonnegative")
        if not isinstance(self.update_interval, int) or self.update_interval < 1:
            raise ValueError("Predetermined update interval must be a positive integer")


class PredeterminedScheduler(AdaptiveScheduler):
    """Use initial ratios for step 1; update after every N completed updates.

    OFF = off_init - n * off_decrement; ON = on_init + n * on_increment;
    SELF = 1 - OFF - ON. Equal increments keep SELF fixed. The whole ramp
    saturates at the first simplex boundary, preserving all three ratios.
    Sampling/broadcasting and optimizer-step counting match adaptive routing.
    """

    def __init__(self, config=None, seed=42):
        super().__init__(seed=seed)
        self.config = config or PredeterminedConfig()

    @property
    def mode_probabilities(self):
        c = self.config
        updates = sum(self.counts.values()) // c.update_interval
        limits = [float(updates)]
        if c.off_decrement:
            limits.append(c.off_init / c.off_decrement)
        if c.on_increment:
            limits.append((1 - c.on_init) / c.on_increment)
        if c.on_increment > c.off_decrement:
            limits.append(c.self_init / (c.on_increment - c.off_decrement))
        progress = min(limits)
        off = max(0.0, c.off_init - progress * c.off_decrement)
        on = min(1.0, c.on_init + progress * c.on_increment)
        # Remove floating point noise at the boundary, then normalize it away.
        ratios = (off, max(0.0, 1 - off - on), on)
        total = sum(ratios)
        return dict(zip(self.MODES, (v / total for v in ratios)))

    def on_evaluation(self, *args, **kwargs):
        raise RuntimeError("Predetermined scheduling must not depend on dev losses")
