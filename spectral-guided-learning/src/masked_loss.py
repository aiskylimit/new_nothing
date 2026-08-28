"""Selective masked training objective (paper Eq. 9).

    L = -(1/Z) * sum_t M_t * log P(y_t | y_<t),   Z = sum_t M_t

Z counts supervised tokens over the whole optimizer step, not per sequence, so every
token gets equal weight regardless of local mask density; under gradient accumulation the
caller passes the full-step count (see MaskedSFTTrainer). Masked positions still run through
the forward pass but contribute no gradient; an all-ones mask reduces this to standard SFT
cross-entropy (asserted by the unit tests).
"""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    denominator: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Cross-entropy over unmasked positions only, normalized by supervised-token count.

    Args:
        logits: (B, T, V) raw model logits (unshifted).
        labels: (B, T) target ids with IGNORE_INDEX at masked/prompt positions (unshifted).
        denominator: Z to divide the summed loss by. Defaults to this batch's own
            supervised-token count; pass a wider count to keep gradient-accumulation
            microbatches on one common normalizer.

    Returns:
        Scalar loss.
    """
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]

    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )

    if denominator is None:
        denominator = (shift_labels != IGNORE_INDEX).sum()
    return token_loss / torch.as_tensor(denominator, device=token_loss.device).float().clamp_min(1.0)
