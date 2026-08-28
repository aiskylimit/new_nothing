"""Analytic loss-gradient capture w.r.t. final hidden states (paper Eq. 1).

g = d(CE)/dh = W_U^T (softmax(W_U h) - onehot(y)) -- closed form, no backward pass needed.
Chunked over target positions to avoid materializing the full (T x V) logits matrix.
"""

import torch


def analytic_hidden_gradients(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    unembedding: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Gradient of per-token CE loss w.r.t. the hidden state that predicts each target.

    Args:
        hidden_states: (N, d) final-layer (post-norm) hidden states, one row per target,
            already shifted so row i is the state predicting target_ids[i].
        target_ids: (N,) target token ids.
        unembedding: (V, d) lm_head weight.
        chunk_size: number of target positions processed at once.

    Returns:
        (N, d) float32 gradient matrix G (paper Eq. 2).
    """
    n_targets = hidden_states.shape[0]
    unembed32 = unembedding.float()
    out = torch.empty(n_targets, unembed32.shape[1], dtype=torch.float32, device=hidden_states.device)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        logits = hidden_states[start:end].float() @ unembed32.T  # (chunk, V)
        probs = torch.softmax(logits, dim=-1)
        # subtract the one-hot target: d(CE)/d(logits) = p - onehot(y)
        probs[torch.arange(end - start, device=probs.device), target_ids[start:end]] -= 1.0
        out[start:end] = probs @ unembed32  # (chunk, V) @ (V, d)
        del logits, probs

    return out


def shift_for_causal_lm(
    last_hidden_state: torch.Tensor, input_ids: torch.Tensor, target_span: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align hidden states with the targets they predict, restricted to a span.

    In a causal LM the state at position i predicts token i+1, so target token at
    absolute position j is predicted by hidden state j-1.

    Args:
        last_hidden_state: (T, d) post-norm final hidden states for one sequence.
        input_ids: (T,) token ids for that sequence.
        target_span: (start, end) absolute positions of supervised tokens (the response).

    Returns:
        (hidden_rows (N, d), target_ids (N,)) with N = end - start.
    """
    start, end = target_span
    if start < 1:
        raise ValueError("target span must start at position >= 1 (needs a preceding token)")
    return last_hidden_state[start - 1 : end - 1], input_ids[start:end]


@torch.no_grad()
def capture_sequence_gradients(
    model,
    input_ids: torch.Tensor,
    target_span: tuple[int, int],
    chunk_size: int = 1024,
    unembedding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward one sequence and return its gradient matrix G over the supervised span.

    Uses model.model(...) to skip the lm_head projection over all positions; unembedding
    is applied chunk-wise inside analytic_hidden_gradients instead. Pass a pre-cast
    float32 unembedding to reuse across a corpus run (optional -- otherwise fetched per
    call, with no numeric difference).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if unembedding is None:
        unembedding = model.get_output_embeddings().weight

    hidden = model.model(input_ids=input_ids).last_hidden_state[0]  # (T, d), post-norm
    rows, targets = shift_for_causal_lm(hidden, input_ids[0], target_span)
    return analytic_hidden_gradients(rows, targets, unembedding, chunk_size=chunk_size)
