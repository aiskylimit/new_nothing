"""L_trans on synthetic hidden states (CPU, no model)."""

import torch
import torch.nn as nn

from sgl.training.collator import MaskedSFTCollator
from sgl.training.transition_loss import (
    HiddenStateCapture,
    TransitionPredictor,
    default_transition_layer,
    find_decoder_layers,
    pool_step_targets,
    shuffled_targets,
    transition_loss,
)

D = 16


def _sequence(step_lengths: list[int], prompt: int = 2, seed: int = 0):
    """Hidden states with one random direction per step plus a strong shared component."""
    torch.manual_seed(seed)
    step_id, step_end, cursor = [-1] * prompt, [], prompt
    for index, length in enumerate(step_lengths):
        step_id += [index] * length
        cursor += length
        step_end.append(cursor - 1)
    step_id += [-1]  # stop token
    hidden = torch.randn(len(step_id), D)
    hidden = hidden + 5.0 * torch.ones(D)  # anisotropy: every step shares a large common vector
    return (
        hidden.requires_grad_(True),
        torch.tensor(step_id),
        torch.tensor(step_end),
        torch.arange(len(step_lengths) - 1),
    )


def test_pooling_removes_the_shared_component():
    hidden, step_id, step_end, _ = _sequence([4, 4, 4])
    z_raw, z_tilde = pool_step_targets(hidden, step_id, 3)
    assert z_raw.shape == (3, D) and not z_raw.requires_grad
    # raw vectors are nearly parallel because of the shared offset; centred ones are not
    raw_cos = nn.functional.cosine_similarity(z_raw[0], z_raw[1], dim=0)
    tilde_cos = nn.functional.cosine_similarity(z_tilde[0], z_tilde[1], dim=0)
    assert raw_cos > 0.9 and tilde_cos < raw_cos
    assert torch.allclose(z_tilde.sum(0), torch.zeros(D), atol=1e-5)
    # a step's own mean, not the whole sequence's
    assert torch.allclose(z_raw[1], hidden[6:10].detach().mean(0))


def test_loss_range_gradient_and_stats():
    hidden, step_id, step_end, pairs = _sequence([4, 5, 6, 3])
    predictor = TransitionPredictor(D, hidden=8)
    loss, stats = transition_loss(hidden, step_id, step_end, pairs, predictor)
    assert 0.0 <= float(loss) <= 2.0
    assert stats.pairs == 3 and -1 <= stats.copy_cos <= 1 and stats.raw_step_cos > 0.9
    assert stats.ztilde_norm > 0

    loss.backward()
    grad = hidden.grad
    # gradient reaches the source positions (last token of steps 0..2) ...
    assert all(grad[int(step_end[i])].abs().sum() > 0 for i in range(3))
    # ... and nothing else: targets are detached, prompt/stop/last step untouched
    source = set(int(step_end[i]) for i in range(3))
    others = [t for t in range(hidden.size(0)) if t not in source]
    assert torch.all(grad[others] == 0)
    assert all(p.grad is not None for p in predictor.parameters())


def test_perfect_predictor_gives_zero_loss():
    hidden, step_id, step_end, pairs = _sequence([4, 4, 4])
    _, z_tilde = pool_step_targets(hidden, step_id, 3)

    class Oracle(nn.Module):
        def forward(self, s):  # ignores s, returns the true next-step target for pairs 0,1
            return z_tilde[1:3]

    loss, stats = transition_loss(hidden, step_id, step_end, pairs, Oracle())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5) and stats.cos > 0.999


def test_no_pairs_returns_zero_that_still_touches_the_predictor():
    hidden, step_id, step_end, _ = _sequence([4])
    predictor = TransitionPredictor(D, hidden=8)
    loss, stats = transition_loss(hidden, step_id, step_end, torch.zeros(0, dtype=torch.long), predictor)
    assert float(loss) == 0.0 and loss.requires_grad and stats.pairs == 0
    loss.backward()
    assert all(p.grad is not None for p in predictor.parameters())


def test_shuffled_targets_never_pick_the_true_next_step():
    torch.manual_seed(0)
    num_steps, pair_src = 6, torch.arange(5)
    for _ in range(200):
        draw = shuffled_targets(pair_src, num_steps)
        assert torch.all(draw != pair_src + 1) and torch.all((0 <= draw) & (draw < num_steps))
    # with 2 steps the only alternative is the source step itself
    assert int(shuffled_targets(torch.tensor([0]), 2)) == 0


def test_capture_hook_takes_once_and_disarms():
    layer = nn.Linear(D, D)
    capture = HiddenStateCapture(layer)
    x = torch.randn(3, D)
    capture.arm()
    out = layer(x)
    assert capture.take() is out
    layer(x)  # disarmed: a re-forward (checkpoint recompute) must not re-capture
    try:
        capture.take()
    except RuntimeError:
        pass
    else:
        raise AssertionError("take() after a disarmed forward should fail")
    capture.remove()


def test_find_decoder_layers_and_default_depth():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(D, D) for _ in range(36)])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    assert len(find_decoder_layers(Outer())) == 36
    assert default_transition_layer(36) == 24


def test_collator_pads_transition_fields_with_minus_one():
    collator = MaskedSFTCollator(pad_token_id=0)
    a = {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1], "step_id": [-1, 0, 0, 1], "step_end": [2, 3],
         "num_steps": 2, "pair_src": [0]}
    b = {"input_ids": [1, 2], "loss_mask": [0, 1], "step_id": [-1, 0], "step_end": [1], "num_steps": 1, "pair_src": []}
    batch = collator([a, b])
    assert batch["step_id"].tolist() == [[-1, 0, 0, 1], [-1, 0, -1, -1]]
    assert batch["step_end"].tolist() == [[2, 3], [1, -1]]
    assert batch["pair_src"].tolist() == [[0], [-1]]
    assert batch["num_steps"].tolist() == [2, 1]
    # plain masked records are untouched
    assert "step_id" not in collator([{"input_ids": [1, 2], "loss_mask": [1, 1]}])


def test_hidden_grad_probe_splits_nll_and_transition_shares():
    """The probe must recover ||dL_NLL/dH|| and ||dL_trans/dH|| from one backward pass."""
    from sgl.training.transition_trainer import _HiddenGradProbe

    torch.manual_seed(1)
    hidden, step_id, step_end, pairs = _sequence([4, 5, 6, 3])
    base = hidden.detach().clone().requires_grad_(True)
    hidden = (base * 1.0).unsqueeze(0)  # [1, T, d] non-leaf, like a captured layer output
    head = nn.Linear(D, 3)
    predictor = TransitionPredictor(D, hidden=8)

    metrics = {}
    probe = _HiddenGradProbe(lambda key, value: metrics.__setitem__(key, value))
    loss_nll = head(hidden).pow(2).mean()  # "layers above" path, touches every position
    loss_trans, _ = transition_loss(
        hidden[0], step_id, step_end, pairs, predictor, source_grad_hook=probe.source_hook(0)
    )
    hidden.register_hook(probe.hidden_hook)
    (loss_nll + 0.3 * loss_trans).backward()

    # reference: separate backward passes on fresh copies
    ref = base.detach().clone().requires_grad_(True)
    head(ref).pow(2).mean().backward()
    g_nll = ref.grad.norm()
    ref2 = base.detach().clone().requires_grad_(True)
    (0.3 * transition_loss(ref2, step_id, step_end, pairs, predictor)[0]).backward()
    g_trans = ref2.grad.norm()

    assert abs(metrics["grad_nll_hidden"] - float(g_nll)) < 1e-4
    assert abs(metrics["grad_trans_hidden"] - float(g_trans)) < 1e-4
    assert abs(metrics["grad_trans_ratio"] - float(g_trans / g_nll)) < 1e-3
