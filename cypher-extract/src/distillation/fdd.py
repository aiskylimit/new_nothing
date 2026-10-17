from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def causal_hidden_state_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask the final valid token to match the template's `tokens[:-1]` input."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence].")
    mask = attention_mask.clone()
    for row in mask:
        positions = torch.nonzero(row, as_tuple=False)
        if positions.numel() > 0:
            row[int(positions[-1].item())] = 0
    return mask


def causal_response_mask(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select hidden states whose next token is an unmasked training label.

    A causal model's logits at position ``t`` predict ``labels[t + 1]``.
    This keeps feature distillation aligned with assistant-only SFT labels:
    user, system, and tool-observation tokens never receive the FDD feature
    loss, while assistant text and function-call serializations do.
    """

    if labels.ndim != 2 or attention_mask.ndim != 2 or labels.shape != attention_mask.shape:
        raise ValueError("labels and attention_mask must have the same two-dimensional shape.")
    mask = torch.zeros_like(attention_mask)
    if labels.shape[1] > 1:
        mask[:, :-1] = labels[:, 1:].ne(-100) & attention_mask[:, :-1].bool()
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not torch.any(mask):
        raise ValueError("FDD received an empty attention mask.")
    return values.masked_fill(~mask, 0.0).sum() / mask.sum()


def soft_label_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    token_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    if mask is None:
        return token_loss.mean()
    return _masked_mean(token_loss, mask)


def fdd_loss(
    student_hidden_states: Sequence[torch.Tensor],
    teacher_hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    student_lm_head: torch.nn.Module,
    teacher_lm_head: torch.nn.Module,
    student_layer_mapping: Sequence[int],
    teacher_layer_mapping: Sequence[int],
) -> torch.Tensor:
    """Feature distribution distillation from the reference FDD trainer.

    Config values are Hugging Face hidden-state indices, not zero-based block
    IDs. For a model with ``n`` blocks, index zero is the embedding output,
    index ``k`` is the representation after block ``k - 1``, and index ``n``
    is the final representation after block ``n - 1`` and the model's final
    norm. Therefore block IDs ``n / 2 - 1`` and ``n - 1`` map to config values
    ``n / 2`` and ``n``. The reference prepended ``None`` to hooked student
    block outputs, producing the same one-position offset.

    Hidden states are compacted to supervised response positions before the
    vocabulary projection. This is mathematically equivalent to the reference
    mask-after-projection formula while avoiding full-vocabulary intermediates
    for prompt and padding tokens.
    """

    if len(student_layer_mapping) != len(teacher_layer_mapping):
        raise ValueError("Student and teacher layer mappings must have equal length.")
    if len(student_layer_mapping) < 2:
        raise ValueError("FDD requires at least two mapped layer pairs.")
    if attention_mask.ndim != 2:
        raise ValueError("FDD attention mask must have shape [batch, sequence].")
    response_mask = attention_mask.bool()
    if not torch.any(response_mask):
        raise ValueError("FDD received an empty attention mask.")

    trajectory_loss: torch.Tensor | None = None
    derivative_loss: torch.Tensor | None = None
    previous_student_logs: torch.Tensor | None = None
    previous_teacher_logs: torch.Tensor | None = None

    for student_index, teacher_index in zip(student_layer_mapping, teacher_layer_mapping, strict=True):
        try:
            student_hidden = student_hidden_states[student_index]
            teacher_hidden = teacher_hidden_states[teacher_index]
        except IndexError as exc:
            raise ValueError(
                f"FDD layer mapping ({student_index}, {teacher_index}) exceeds available hidden states "
                f"({len(student_hidden_states)}, {len(teacher_hidden_states)})."
            ) from exc

        if student_hidden.shape[:2] != response_mask.shape or teacher_hidden.shape[:2] != response_mask.shape:
            raise ValueError("FDD hidden states and attention mask must share batch/sequence dimensions.")
        student_hidden = student_hidden[response_mask]
        teacher_hidden = teacher_hidden[response_mask]

        student_hidden_logits = student_lm_head(student_hidden)
        with torch.no_grad():
            teacher_hidden_logits = teacher_lm_head(teacher_hidden)
        shared_vocab_size = min(student_hidden_logits.shape[-1], teacher_hidden_logits.shape[-1])
        if shared_vocab_size <= 0:
            raise ValueError("Student and teacher vocabularies must be non-empty.")
        student_hidden_logits = student_hidden_logits[..., :shared_vocab_size]
        teacher_hidden_logits = teacher_hidden_logits[..., :shared_vocab_size]
        current_trajectory_loss = soft_label_distillation_loss(student_hidden_logits, teacher_hidden_logits, None)
        trajectory_loss = (
            current_trajectory_loss if trajectory_loss is None else trajectory_loss + current_trajectory_loss
        )

        student_logs = F.log_softmax(student_hidden_logits, dim=-1)
        teacher_logs = F.log_softmax(teacher_hidden_logits, dim=-1)
        if previous_student_logs is not None and previous_teacher_logs is not None:
            student_delta = student_logs - previous_student_logs
            teacher_delta = teacher_logs - previous_teacher_logs
            cosine_loss = 1.0 - F.cosine_similarity(student_delta, teacher_delta, dim=-1, eps=1e-5)
            current_derivative_loss = cosine_loss.mean()
            derivative_loss = (
                current_derivative_loss if derivative_loss is None else derivative_loss + current_derivative_loss
            )

        previous_student_logs = student_logs
        previous_teacher_logs = teacher_logs

    pair_count = len(student_layer_mapping)
    if trajectory_loss is None or derivative_loss is None:  # guarded by the pair-count validation above
        raise RuntimeError("FDD loss accumulation failed.")
    return trajectory_loss / pair_count + derivative_loss / (pair_count - 1)
