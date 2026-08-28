"""Verifies the analytic gradient formula against autograd on a tiny randomly
initialized Qwen3 model (runs offline on CPU)."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from gradient_utils import analytic_hidden_gradients, capture_sequence_gradients, shift_for_causal_lm


@pytest.fixture(scope="module")
def tiny_model():
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval().float()
    return model


def test_analytic_matches_autograd(tiny_model):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (1, 24))
    span = (8, 24)  # supervised "response" span

    analytic = capture_sequence_gradients(tiny_model, input_ids, span, chunk_size=5)

    # autograd reference: dL_t/dh for the summed per-token losses
    hidden = tiny_model.model(input_ids=input_ids).last_hidden_state[0].detach().requires_grad_(True)
    rows, targets = shift_for_causal_lm(hidden, input_ids[0], span)
    logits = tiny_model.lm_head(rows)
    torch.nn.functional.cross_entropy(logits, targets, reduction="sum").backward()
    reference = hidden.grad[span[0] - 1 : span[1] - 1]

    assert analytic.shape == reference.shape
    assert torch.allclose(analytic, reference, rtol=1e-4, atol=1e-6)


def test_chunking_does_not_change_result(tiny_model):
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (1, 20))
    single_chunk = capture_sequence_gradients(tiny_model, input_ids, (4, 20), chunk_size=1024)
    many_chunks = capture_sequence_gradients(tiny_model, input_ids, (4, 20), chunk_size=3)
    assert torch.allclose(single_chunk, many_chunks, atol=1e-6)


def test_hidden_states_are_post_norm_so_recomputed_logits_match(tiny_model):
    """capture_sequence_gradients relies on last_hidden_state being the exact lm_head input."""
    torch.manual_seed(2)
    input_ids = torch.randint(0, 64, (1, 12))
    hidden = tiny_model.model(input_ids=input_ids).last_hidden_state
    recomputed = hidden @ tiny_model.get_output_embeddings().weight.T
    assert torch.allclose(recomputed, tiny_model(input_ids=input_ids).logits, atol=1e-4)


def test_gradient_of_confident_correct_prediction_is_small():
    """Sanity on the formula itself: p ~ onehot(y) => gradient ~ 0."""
    unembedding = torch.eye(8) * 20.0  # strongly separates classes
    hidden = torch.zeros(1, 8)
    hidden[0, 3] = 1.0  # predicts class 3 with near-certainty
    gradient = analytic_hidden_gradients(hidden, torch.tensor([3]), unembedding)
    assert gradient.abs().max() < 1e-3


def test_shift_rejects_span_starting_at_zero():
    with pytest.raises(ValueError):
        shift_for_causal_lm(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), (0, 4))
