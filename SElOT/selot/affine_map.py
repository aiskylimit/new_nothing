"""Per-layer affine steering map, shared by SELOT and every baseline.

Each layer stores the map T_l(h) = A1 h + b1 that full steering (alpha = 1) applies, as its displacement

    T_l(h) - h = s * h + D h + U (V^T h) + b0          (every term optional)

so that at strength alpha

    h <- A h + b,   A = I + alpha (diag(s) + D + U V^T),   b = alpha * b0.

Special cases:
    vector shift (difference in means, Surgical, ...):   only b0          ->  A = I,         b = alpha r
    AlphaSteer form (h <- h + lambda h Delta):          D = Delta^T      ->  A = I + l D,   b = 0
    SELOT / PCA-OT (rank-k field):                      U, V, b0
    Linear-AcT (per-neuron 1-D maps):                   s, b0
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

FORMAT = "selot-affine-v1"
_TERMS = ("s", "D", "U", "V", "b0")


@dataclass
class LayerMap:
    s: Optional[torch.Tensor] = None     # d       diagonal term
    D: Optional[torch.Tensor] = None     # d x d   dense term
    U: Optional[torch.Tensor] = None     # d x r   low-rank term U V^T
    V: Optional[torch.Tensor] = None     # d x r
    b0: Optional[torch.Tensor] = None    # d       bias
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)  # e.g. P, mu_pool, for analysis only

    def __post_init__(self):
        if (self.U is None) != (self.V is None):
            raise ValueError("U and V must be given together")

    def terms(self):
        return {k: getattr(self, k) for k in _TERMS if getattr(self, k) is not None}

    def to(self, device=None, dtype=torch.float32) -> "LayerMap":
        move = lambda t: None if t is None else t.to(device=device, dtype=dtype)
        return LayerMap(**{k: move(getattr(self, k)) for k in _TERMS},
                        extras={k: move(v) for k, v in self.extras.items()})

    def displacement(self, h: torch.Tensor) -> torch.Tensor:
        """T(h) - h for h of shape (..., d), computed in float32 (h is cast if needed)."""
        h = h.to(torch.float32)
        out = torch.zeros_like(h)
        if self.s is not None:
            out = out + h * self.s
        if self.D is not None:
            out = out + h @ self.D.T
        if self.U is not None:
            out = out + (h @ self.V) @ self.U.T
        if self.b0 is not None:
            out = out + self.b0
        return out

    def apply(self, h: torch.Tensor, alpha: float) -> torch.Tensor:
        """A h + b in float32."""
        return h.to(torch.float32) + alpha * self.displacement(h)

    def dense(self, alpha: float, d: int):
        """Explicit (A, b) at strength alpha (tests and small models only: A is d x d)."""
        dev = next(iter(self.terms().values())).device
        a = torch.eye(d, device=dev)
        if self.s is not None:
            a = a + alpha * torch.diag(self.s)
        if self.D is not None:
            a = a + alpha * self.D
        if self.U is not None:
            a = a + alpha * self.U @ self.V.T
        b = alpha * self.b0 if self.b0 is not None else torch.zeros(d, device=dev)
        return a, b


@dataclass
class AffineMap:
    method: str
    hidden_size: int
    layers: Dict[int, LayerMap]
    meta: dict = field(default_factory=dict)

    def to(self, device=None) -> "AffineMap":
        return AffineMap(self.method, self.hidden_size, {l: m.to(device) for l, m in self.layers.items()}, self.meta)

    def save(self, path):
        payload = {"format": FORMAT, "method": self.method, "hidden_size": self.hidden_size, "meta": self.meta,
                   "layers": {int(l): {**{k: v.cpu() for k, v in m.terms().items()},
                                       "extras": {k: v.cpu() for k, v in m.extras.items()}}
                              for l, m in self.layers.items()}}
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device=None) -> "AffineMap":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        layers = {}
        for l, entry in payload["layers"].items():
            entry = dict(entry)
            extras = entry.pop("extras", {})
            layers[int(l)] = LayerMap(**entry, extras=extras)
        return cls(payload["method"], payload["hidden_size"], layers, payload.get("meta", {})).to(device)
