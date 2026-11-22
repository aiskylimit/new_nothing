"""Forward pre-hooks that apply h <- A h + b to the residual stream at the INPUT of decoder layer l.

That is the same tensor as `output_hidden_states[l]` and the one activations are extracted from
(selot.activations), so the map acts exactly where it was fitted. One hook per steered layer, any
architecture with a `layers` ModuleList (Llama, Qwen2, Gemma2, ...): no model subclassing.

Positions (`positions`):
    prompt      default. Every prompt position t gets its own A h_t + b, at prefill only; generated tokens
                are not steered.
    alphasteer  AlphaSteer's rule (baseline only): one displacement computed from the LAST prompt token,
                added to every prompt position, at prefill only.
    all         every position, prompt and generated (proposal Eq. 11 as written; ablation).

Prompts are left padded, so prompt positions are 0..prompt_len-1 and the last prompt token is at
prompt_len-1. A forward with fewer than prompt_len positions is a cached decode step and is left alone
under prompt/alphasteer; a forward over more than prompt_len positions (use_cache=False) only steers its
first prompt_len positions, so generated tokens are never steered under those two modes.
"""

from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch

from selot.affine_map import AffineMap

POSITIONS = ("prompt", "alphasteer", "all")


def decoder_layers(model) -> torch.nn.ModuleList:
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers",
                 "model.language_model.layers"):
        obj = model
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise ValueError(f"cannot find the decoder layer list of {type(model).__name__}")


def _get_hidden(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _set_hidden(args, kwargs, hidden):
    if args:
        return (hidden,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = hidden
    return args, kwargs


class Steerer:
    """Holds the map, the per-layer strengths and the current prompt length; installs the hooks."""

    def __init__(self, model, amap: AffineMap, positions: str = "prompt", layers: Optional[Iterable[int]] = None):
        if positions not in POSITIONS:
            raise ValueError(f"positions must be one of {POSITIONS}")
        self.model = model
        self.positions = positions
        self.layers = sorted(amap.layers) if layers is None else sorted(layers)
        missing = set(self.layers) - set(amap.layers)
        if missing:
            raise ValueError(f"no fitted map for layers {sorted(missing)}")
        blocks = decoder_layers(model)
        self.maps = {l: amap.layers[l].to(next(blocks[l].parameters()).device) for l in self.layers}
        self.alpha: Dict[int, float] = {l: 0.0 for l in self.layers}
        self.prompt_len: Optional[int] = None
        self._handles = []

    def set_alpha(self, alpha):
        """alpha: one float for every steered layer, or {layer: float}."""
        for l in self.layers:
            self.alpha[l] = float(alpha[l] if isinstance(alpha, dict) else alpha)

    def _hook(self, layer):
        def hook(module, args, kwargs):
            alpha = self.alpha[layer]
            if alpha == 0.0:
                return None
            h = _get_hidden(args, kwargs)
            lmap = self.maps[layer]
            t = h.shape[1]
            if self.positions == "all":
                new = h + (alpha * lmap.displacement(h)).to(h.dtype)
            else:
                if self.prompt_len is None:
                    raise RuntimeError("Steerer.prompt_len is not set (use Steerer.prompt(...) around generate)")
                if t < self.prompt_len:                       # cached decode step: generated token
                    return None
                n = self.prompt_len
                if self.positions == "prompt":
                    shift = alpha * lmap.displacement(h[:, :n])
                else:                                          # alphasteer: from the last prompt token
                    shift = (alpha * lmap.displacement(h[:, n - 1])).unsqueeze(1).expand(-1, n, -1)
                new = h.clone()
                new[:, :n] = h[:, :n] + shift.to(h.dtype)
            return _set_hidden(args, kwargs, new)
        return hook

    def attach(self):
        blocks = decoder_layers(self.model)
        for l in self.layers:
            self._handles.append(blocks[l].register_forward_pre_hook(self._hook(l), with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextmanager
    def prompt(self, prompt_len: int):
        """Context for one batch: the (left-padded) prompt length of the input_ids passed to generate."""
        self.prompt_len = prompt_len
        try:
            yield self
        finally:
            self.prompt_len = None

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()


@contextmanager
def capture_layer_inputs(model, layers: Iterable[int]):
    """Records the input hidden state of each listed decoder layer (after any earlier-registered hook)."""
    store: Dict[int, torch.Tensor] = {}
    blocks = decoder_layers(model)
    handles = []
    for l in layers:
        def hook(module, args, kwargs, l=l):
            store[l] = _get_hidden(args, kwargs).detach()
        handles.append(blocks[l].register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
