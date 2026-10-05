"""Next-step representation prediction (L_trans): predictor, targets, loss, diagnostics.

    L_trans = (1/|P|) sum_{(i,i+1) in P} [1 - cos(f(s_i), z~_{i+1})]

s_i is the hidden state at the last token of step i at one decoder layer (causal, so it has seen
nothing of step i+1; gradient flows through it into LoRA). z_{i+1} is the mean-pooled hidden state
of step i+1 at the same layer, detached, with the per-sequence mean of all step vectors removed
(LLM hidden states are anisotropic: without this a predictor learns the sequence's common
direction and the loss drops while learning nothing) and L2-normalized. Everything runs in fp32.

Framework-agnostic: no Trainer/DeepSpeed here, so it is unit-testable on CPU.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionPredictor(nn.Module):
    """f(s) = W2 GELU(W1 LN(s)); d -> hidden -> d. Trained fully (own param group), dropped at
    inference. Default init on purpose: a zero W2 makes the cosine undefined at step 0."""

    def __init__(self, d_model: int, hidden: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # DeepSpeed bf16 casts every submodule (this one included) to bf16 while the source
        # states arrive as fp32; follow the weights' dtype and hand back fp32 for the cosine.
        s = s.to(self.up.weight.dtype)
        return self.down(F.gelu(self.up(self.norm(s)))).float()


@dataclass
class TransitionStats:
    """Diagnostics of one sequence (all plain floats; `pairs` = 0 means nothing was scored)."""

    pairs: int = 0
    cos: float = 0.0  # mean cos(f(s_i), z~_{i+1}) -- the quantity the loss drives up
    copy_cos: float = 0.0  # mean cos(z~_i, z~_{i+1}): what "predict next = current" would score
    raw_step_cos: float = 0.0  # mean pairwise cos of raw z (before mean removal); -> 1 = collapse
    ztilde_norm: float = 0.0  # mean ||z~|| before normalization; -> 0 = step-specific part vanishing
    extra: dict = field(default_factory=dict)


def pool_step_targets(
    hidden: torch.Tensor, step_id: torch.Tensor, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool `hidden` [T, d] per step and remove the per-sequence mean.

    Returns (z_raw [K, d], z_tilde [K, d]); both detached fp32, z_tilde not yet normalized.
    """
    hidden = hidden.detach().float()
    valid = step_id >= 0
    sums = hidden.new_zeros(num_steps, hidden.size(-1))
    sums.index_add_(0, step_id[valid], hidden[valid])
    counts = torch.bincount(step_id[valid], minlength=num_steps).clamp_min(1).unsqueeze(-1)
    z_raw = sums / counts.to(hidden.dtype)
    z_tilde = z_raw - z_raw.mean(0, keepdim=True)
    return z_raw, z_tilde


def _mean_offdiag_cos(z: torch.Tensor) -> float:
    if z.size(0) < 2:
        return 0.0
    unit = F.normalize(z, dim=-1)
    gram = unit @ unit.T
    k = z.size(0)
    return float((gram.sum() - gram.diagonal().sum()) / (k * (k - 1)))


def shuffled_targets(pair_src: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Control: for source step i pick a random j != i+1 from the same sequence (j == i allowed).

    Keeps topic, regularization strength and pair count identical to the real objective and only
    destroys the transition order; requires num_steps >= 2.
    """
    draws = torch.randint(0, num_steps - 1, pair_src.shape, device=pair_src.device)
    # skip over i+1: values >= i+1 shift up by one, so the range covers [0, K) \ {i+1}
    return draws + (draws >= pair_src + 1).long()


def transition_loss(
    hidden: torch.Tensor,
    step_id: torch.Tensor,
    step_end: torch.Tensor,
    pair_src: torch.Tensor,
    predictor: nn.Module,
    shuffle_targets: bool = False,
    source_grad_hook=None,
) -> tuple[torch.Tensor, TransitionStats]:
    """L_trans for one sequence.

    Args:
        hidden: [T, d] hidden state of the chosen layer, with graph (source positions keep grad).
        step_id: [T] long, -1 outside the CoT.
        step_end: [K] long, absolute index of each step's last token.
        pair_src: [P] long, source step indices of the kept transitions.
        predictor: TransitionPredictor (fp32).
        shuffle_targets: replace z~_{i+1} by a random other step of the same sequence (control).
        source_grad_hook: optional fn(positions [P], grad [P, d]) run during backward with the
            gradient reaching the source states -- only L_trans flows through them, so this is
            the transition term's contribution to dL/dH at those positions.

    Returns:
        (loss, stats). With no scorable pair the loss is a zero that still touches the predictor,
        so every rank/microbatch produces a gradient entry for its params under ZeRO.
    """
    num_steps = int(step_end.numel())
    if pair_src.numel() == 0 or num_steps < 2:
        zero = predictor(hidden[:1].float()).sum() * 0.0
        return zero, TransitionStats()

    z_raw, z_tilde = pool_step_targets(hidden, step_id, num_steps)
    ztilde_norm = z_tilde.norm(dim=-1)
    targets = F.normalize(z_tilde, dim=-1)

    tgt_index = shuffled_targets(pair_src, num_steps) if shuffle_targets else pair_src + 1
    positions = step_end[pair_src]
    source = hidden[positions].float()  # [P, d], gradient flows into LoRA
    if source_grad_hook is not None:
        source.register_hook(lambda grad: source_grad_hook(positions, grad.detach()))
    pred = F.normalize(predictor(source), dim=-1)
    cos = (pred * targets[tgt_index]).sum(-1)
    loss = (1.0 - cos).mean()

    with torch.no_grad():
        copy_cos = (targets[pair_src] * targets[pair_src + 1]).sum(-1).mean()
        stats = TransitionStats(
            pairs=int(pair_src.numel()),
            cos=float(cos.mean()),
            copy_cos=float(copy_cos),
            raw_step_cos=_mean_offdiag_cos(z_raw),
            ztilde_norm=float(ztilde_norm.mean()),
        )
    return loss, stats


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The decoder-layer ModuleList of a (possibly PEFT-wrapped) HF causal LM.

    PEFT: model.base_model.model is the HF model; HF: model.model.layers (Qwen/Llama layout).
    Falls back to the first ModuleList attribute named `layers` found by a module walk.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", None)
    layers = getattr(inner, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return layers
    for name, module in base.named_modules():
        if name.endswith("layers") and isinstance(module, nn.ModuleList):
            return module
    raise ValueError("could not locate the decoder layers (expected <model>.model.layers)")


def default_transition_layer(num_layers: int) -> int:
    """0-based index into the decoder layers at ~2/3 depth (24 of 36 for Qwen3-8B). The final
    layers are tied to the next-token distribution; mid-depth keeps more of the step's semantics."""
    return round(2 * num_layers / 3)


class HiddenStateCapture:
    """Forward hook on one decoder layer: keeps that layer's output for the current forward.

    Cheaper than output_hidden_states=True (which pins every layer's [T, d] output). `take()`
    returns the tensor once and disarms the hook, so the re-forward gradient checkpointing runs
    during backward does not overwrite it or pin a second copy.
    """

    def __init__(self, layer: nn.Module):
        self._hidden = None
        self._armed = False
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self._armed:
            self._hidden = output[0] if isinstance(output, tuple) else output
            self._armed = False

    def arm(self) -> None:
        self._hidden = None
        self._armed = True

    def take(self) -> torch.Tensor:
        if self._hidden is None:
            raise RuntimeError("no hidden state captured: arm() must precede the forward pass")
        hidden, self._hidden = self._hidden, None
        return hidden

    def remove(self) -> None:
        self._handle.remove()
