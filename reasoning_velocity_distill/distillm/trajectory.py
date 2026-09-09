import math

import torch


def find_step_spans(input_ids, attention_mask, labels, marker_ids, max_steps):
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected input_ids and attention_mask with shape [B, L]")
    if labels.shape != input_ids.shape:
        raise ValueError("Labels must have the same shape as input_ids")
    if marker_ids.ndim != 1 or marker_ids.numel() == 0:
        raise ValueError("marker_ids must be a nonempty 1D tensor")
    max_steps = int(max_steps)
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    spans = input_ids.new_full((input_ids.shape[0], max_steps, 2), -1)
    if max_steps == 0 or input_ids.shape[1] == 0:
        return spans
    marker_ids = marker_ids.to(input_ids.device)
    marker_length = marker_ids.numel()
    valid_lengths = attention_mask.sum(dim=-1)
    supervised = labels != -100
    response_starts = supervised.long().argmax(dim=-1) + 1
    response_starts = torch.where(
        supervised.any(dim=-1), response_starts, valid_lengths
    )
    if input_ids.shape[1] < marker_length:
        valid = response_starts < valid_lengths
        spans[:, 0, 0] = torch.where(valid, response_starts, -1)
        spans[:, 0, 1] = torch.where(valid, valid_lengths, -1)
        return spans
    marker_starts = torch.arange(
        input_ids.shape[1] - marker_length + 1, device=input_ids.device
    )
    matches = (input_ids.unfold(1, marker_length, 1) == marker_ids).all(dim=-1)
    matches &= marker_starts[None, :] >= response_starts[:, None]
    matches &= marker_starts[None, :] + marker_length <= valid_lengths[:, None]

    # Select markers left-to-right. Keeping the cursor on device avoids a CPU/GPU
    # synchronization for every trajectory and also rejects overlapping matches.
    cursor = response_starts
    step_counts = torch.zeros_like(response_starts)
    sentinel = marker_starts.numel()
    for _ in range(max_steps):
        eligible = matches & (marker_starts[None, :] >= cursor[:, None])
        next_marker = torch.where(eligible, marker_starts, sentinel).amin(dim=-1)
        valid = next_marker < sentinel
        if not valid.any():
            break
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        valid_indices = batch_indices[valid]
        valid_steps = step_counts[valid]
        marker_end = next_marker + marker_length
        spans[valid_indices, valid_steps, 0] = cursor[valid]
        spans[valid_indices, valid_steps, 1] = marker_end[valid]
        step_counts = step_counts + valid.long()
        cursor = torch.where(valid, marker_end, cursor)

    tail_valid = (cursor < valid_lengths) & (step_counts < max_steps)
    if tail_valid.any():
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        tail_indices = batch_indices[tail_valid]
        tail_steps = step_counts[tail_valid]
        spans[tail_indices, tail_steps, 0] = cursor[tail_valid]
        spans[tail_indices, tail_steps, 1] = valid_lengths[tail_valid]
    return spans


def pool_steps(hidden_states, step_spans, pooling="mean"):
    if pooling not in ("mean", "last_token"):
        raise ValueError(f"Unknown step pooling: {pooling}")
    if hidden_states.ndim != 3 or step_spans.ndim != 3 or step_spans.shape[-1] != 2:
        raise ValueError("Expected hidden states [B, L, D] and step spans [B, T, 2]")
    if hidden_states.shape[0] != step_spans.shape[0]:
        raise ValueError("Hidden states and spans must have the same batch size")
    start, end = step_spans.unbind(-1)
    valid = (start >= 0) & (end > start) & (end <= hidden_states.shape[1])
    start = start.masked_fill(~valid, 0)
    end = end.masked_fill(~valid, 0)
    hidden = hidden_states.float()
    batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    if pooling == "last_token":
        pooled = hidden[batch, (end - 1).clamp_min(0)]
    else:
        prefix = torch.cat(
            (
                hidden.new_zeros(hidden.shape[0], 1, hidden.shape[2]),
                hidden.cumsum(dim=1),
            ),
            dim=1,
        )
        pooled = (prefix[batch, end] - prefix[batch, start]) / (end - start).clamp_min(
            1
        )[..., None]
    return pooled.masked_fill(~valid[..., None], 0), valid


def _relative_magnitudes(magnitudes, mask, normalization, eps):
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (magnitudes * mask).sum(dim=-1, keepdim=True) / count
    if normalization == "mean":
        result = magnitudes / (mean + eps)
    elif normalization == "zscore":
        centered = (magnitudes - mean).masked_fill(~mask, 0)
        # Population std; vector_norm has a finite gradient at an all-zero input.
        std = torch.linalg.vector_norm(centered, dim=-1, keepdim=True) / count.sqrt()
        result = centered / (std + eps)
    else:
        raise ValueError(f"Unknown magnitude normalization: {normalization}")
    return result.masked_fill(~mask, 0)


def _trajectory_average(values, mask):
    counts = mask.sum(dim=-1)
    per_sample = values.masked_fill(~mask, 0).sum(dim=-1) / counts.clamp_min(1)
    eligible = counts > 0
    return per_sample.sum() / eligible.sum().clamp_min(1)


def reasoning_velocity_loss(
    student_hidden,
    teacher_hidden,
    student_spans,
    teacher_spans=None,
    pooling="mean",
    normalization="zscore",
    eps=1e-6,
):
    """Compare velocities between consecutive response steps."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if teacher_spans is None:
        teacher_spans = student_spans
    if student_spans.shape != teacher_spans.shape:
        raise ValueError("Teacher/student spans must align on batch and step indices")
    student, s_mask = pool_steps(student_hidden, student_spans, pooling)
    teacher, t_mask = pool_steps(teacher_hidden.detach(), teacher_spans, pooling)
    return velocity_loss_from_steps(
        student, teacher, s_mask, t_mask, normalization, eps
    )


def velocity_loss_from_steps(
    student, teacher, s_mask, t_mask, normalization="zscore", eps=1e-6
):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (
        student.shape[:2] != teacher.shape[:2]
        or s_mask.shape != student.shape[:2]
        or t_mask.shape != teacher.shape[:2]
    ):
        raise ValueError(
            "Representations and masks must align on batch and step indices"
        )
    student = student.float()
    teacher = teacher.detach().float()
    step_mask = s_mask & t_mask
    velocity_mask = step_mask[:, 1:] & step_mask[:, :-1]
    student_delta = student[:, 1:] - student[:, :-1]
    teacher_delta = teacher[:, 1:] - teacher[:, :-1]
    student_mag = torch.linalg.vector_norm(student_delta, dim=-1)
    teacher_mag = torch.linalg.vector_norm(teacher_delta, dim=-1)
    student_z = _relative_magnitudes(student_mag, velocity_mask, normalization, eps)
    teacher_z = _relative_magnitudes(teacher_mag, velocity_mask, normalization, eps)
    mag_loss = _trajectory_average((student_z - teacher_z).square(), velocity_mask)

    student_dir = student_delta / (student_mag[..., None] + eps)
    teacher_dir = teacher_delta / (teacher_mag[..., None] + eps)
    student_gram = student_dir @ student_dir.transpose(-1, -2)
    teacher_gram = teacher_dir @ teacher_dir.transpose(-1, -2)
    pair_mask = (velocity_mask[:, :, None] & velocity_mask[:, None, :]).triu(diagonal=1)
    gram_loss = _trajectory_average(
        (student_gram - teacher_gram).square().flatten(1), pair_mask.flatten(1)
    )
    return mag_loss, gram_loss
