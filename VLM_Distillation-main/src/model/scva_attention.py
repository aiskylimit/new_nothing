"""Sparse response-to-vision attention capture used by SCVA."""

from __future__ import annotations

import math
from typing import Optional

import torch


def resolve_layer_pairs(
    n_student_layers: int,
    n_teacher_layers: int,
    n_pairs: int = 4,
    low_pct: float = 0.4,
    high_pct: float = 0.7,
) -> list[tuple[int, int]]:
    """Resolve decoder-layer pairs once, before training starts."""
    n_s = int(n_student_layers)
    n_t = int(n_teacher_layers)
    if n_s <= 0 or n_t <= 0:
        return []
    if not 0.0 <= low_pct <= high_pct <= 1.0:
        raise ValueError(
            "SCVA layer percentages must satisfy 0 <= low_pct <= high_pct <= 1; "
            f"got low_pct={low_pct}, high_pct={high_pct}."
        )

    lo = max(0, min(int(math.floor(low_pct * n_s)), n_s - 1))
    hi = max(lo, min(int(math.floor(high_pct * n_s)), n_s - 1))
    n_pairs = max(int(n_pairs), 1)
    if n_pairs == 1 or lo == hi:
        student_indices = [lo]
    else:
        step = (hi - lo) / (n_pairs - 1)
        student_indices = sorted({min(int(round(lo + i * step)), hi) for i in range(n_pairs)})

    ratio = n_t / n_s
    return [(s_idx, min(int(round(s_idx * ratio)), n_t - 1)) for s_idx in student_indices]


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


def _selected_mask_rows(
    attention_mask: torch.Tensor,
    batch_idx: int,
    query_indices: torch.Tensor,
    key_indices: torch.Tensor,
) -> torch.Tensor:
    batch_pos = 0 if attention_mask.shape[0] == 1 else batch_idx
    mask = attention_mask[batch_pos]
    if mask.ndim == 3:  # [H|1, Q, K]
        return mask[:, query_indices][:, :, key_indices]
    if mask.ndim == 2:  # [Q, K]
        return mask[query_indices][:, key_indices].unsqueeze(0)
    raise ValueError(f"Unsupported SCVA attention-mask shape: {tuple(attention_mask.shape)}")


def capture_response_to_vision_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
) -> list[torch.Tensor] | None:
    """Capture [response, vision] distributions without constructing [L, L]."""
    if not getattr(module, "_scva_capture_enabled", False):
        return None

    response_mask = getattr(module, "_scva_response_mask", None)
    vision_mask = getattr(module, "_scva_vision_mask", None)
    if not torch.is_tensor(response_mask) or not torch.is_tensor(vision_mask):
        return None

    batch_size, _n_heads, query_len, _head_dim = query_states.shape
    key_states = _repeat_kv(key_states, int(module.num_key_value_groups))
    key_len = key_states.shape[-2]
    if response_mask.shape[0] != batch_size or vision_mask.shape[0] != batch_size:
        raise ValueError("SCVA masks and attention tensors have different batch sizes.")

    response_mask = response_mask[:, -query_len:].to(query_states.device, dtype=torch.bool)
    vision_mask = vision_mask[:, -key_len:].to(query_states.device, dtype=torch.bool)
    captures: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        response_indices = response_mask[batch_idx].nonzero(as_tuple=True)[0]
        vision_indices = vision_mask[batch_idx].nonzero(as_tuple=True)[0]
        if response_indices.numel() == 0 or vision_indices.numel() == 0:
            captures.append(query_states.new_zeros((response_indices.numel(), vision_indices.numel())))
            continue

        query = query_states[batch_idx, :, response_indices, :]
        key = key_states[batch_idx, :, vision_indices, :]
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scaling)

        if torch.is_tensor(attention_mask):
            mask_rows = _selected_mask_rows(
                attention_mask, batch_idx, response_indices, vision_indices
            ).to(scores.device)
            if mask_rows.dtype == torch.bool:
                scores = scores.masked_fill(~mask_rows, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask_rows.to(scores.dtype)
        else:
            query_positions = response_indices + max(key_len - query_len, 0)
            causal = vision_indices.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(~causal.unsqueeze(0), torch.finfo(scores.dtype).min)

        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        visual = probabilities.mean(dim=0)
        captures.append(visual)

    return captures
