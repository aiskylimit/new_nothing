"""NLL theo step / theo expert + step-weighted loss (spec §4.2).

Quy ước alignment (rất dễ sai, đọc kỹ):
    ``logits[:, t, :]`` dự đoán token ``input_ids[:, t + 1]``.
Mọi hàm ở đây nhận span token trong toạ độ ``input_ids`` và tự shift.
Token ở vị trí 0 không có logit dự đoán nên không bao giờ vào loss.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Span = Tuple[int, int]


def token_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    """log p(y_t | y_<t) cho từng vị trí.

    Args:
        logits: ``(T, V)`` — logits của một sequence (batch đã squeeze).
        input_ids: ``(T,)``.
        chunk_size: Kích thước chunk tính cross-entropy để tránh peak VRAM fp32 trên sequence dài.

    Returns:
        ``(T,)``; phần tử 0 luôn bằng 0 (không có dự đoán cho token đầu).
    """
    if logits.dim() == 3:
        if logits.size(0) != 1:
            raise ValueError("token_logprobs chỉ nhận batch size 1")
        logits = logits[0]
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    T_seq = logits.size(0) - 1
    if T_seq <= chunk_size:
        lp = -F.cross_entropy(logits[:-1].float(), input_ids[1:], reduction="none")
    else:
        lp_chunks = []
        for c_start in range(0, T_seq, chunk_size):
            c_end = min(c_start + chunk_size, T_seq)
            c_lp = -F.cross_entropy(
                logits[c_start:c_end].float(),
                input_ids[c_start + 1 : c_end + 1],
                reduction="none",
            )
            lp_chunks.append(c_lp)
        lp = torch.cat(lp_chunks, dim=0)
    return torch.cat([lp.new_zeros(1), lp], dim=0)


def step_nll_from_logprobs(
    logprobs: torch.Tensor, step_spans: Sequence[Span]
) -> torch.Tensor:
    """L_i = -(1/T_i) * sum_{t in step i} log p(y_t).

    ``logprobs`` là output của :func:`token_logprobs` (đã align sang toạ độ
    ``input_ids``). Span rỗng hoặc chỉ chứa vị trí 0 trả về 0.
    """
    out = logprobs.new_zeros(len(step_spans))
    for i, (start, end) in enumerate(step_spans):
        start = max(start, 1)  # vị trí 0 không có logprob
        if end <= start:
            continue
        out[i] = -logprobs[start:end].mean()
    return out


def step_nll(
    logits: torch.Tensor, input_ids: torch.Tensor, step_spans: Sequence[Span]
) -> torch.Tensor:
    """Tiện lợi: logits + input_ids -> vector NLL ``(N,)``."""
    return step_nll_from_logprobs(token_logprobs(logits, input_ids), step_spans)


def step_token_counts(step_spans: Sequence[Span]) -> List[int]:
    """T_i thực sự vào loss (đã trừ vị trí 0 nếu span chạm đầu sequence)."""
    return [max(end - max(start, 1), 0) for start, end in step_spans]


# --------------------------------------------------------------------------- #
# Step-weighted loss
# --------------------------------------------------------------------------- #
def weighted_step_loss(
    step_losses: torch.Tensor,
    step_lengths: Sequence[int],
    weights: Optional[torch.Tensor] = None,
    reduction: str = "token_mean",
) -> torch.Tensor:
    """Gộp NLL theo step thành một scalar loss.

    Args:
        step_losses: ``(N,)`` — NLL length-normalized của từng step.
        step_lengths: ``(N,)`` — T_i, dùng cho reduction ``token_mean``.
        weights: ``(N,)`` — w_i; ``None`` nghĩa là w_i = 1.
        reduction:
            ``token_mean`` (mặc định): ``sum_i w_i T_i L_i / sum_i w_i T_i``.
                Với w = 1 nó **bằng đúng** mean NLL trên mọi token có label →
                đây là lý do ablation ``uniform`` phải trùng Vanilla SFT (§14).
            ``step_mean``: ``sum_i w_i L_i / sum_i w_i``.
            ``sum``: ``sum_i w_i L_i`` — đúng công thức (17) trong proposal,
                nhưng scale loss phụ thuộc N nên không dùng làm mặc định.
    """
    if weights is None:
        weights = step_losses.new_ones(step_losses.shape)
    weights = weights.to(step_losses.dtype).to(step_losses.device)
    if weights.shape != step_losses.shape:
        raise ValueError(f"weights {tuple(weights.shape)} != losses {tuple(step_losses.shape)}")

    if reduction == "sum":
        return (weights * step_losses).sum()
    if reduction == "step_mean":
        denom = weights.sum().clamp_min(1e-8)
        return (weights * step_losses).sum() / denom
    if reduction == "token_mean":
        lengths = torch.as_tensor(
            list(step_lengths), dtype=step_losses.dtype, device=step_losses.device
        )
        coef = weights * lengths
        denom = coef.sum().clamp_min(1e-8)
        return (coef * step_losses).sum() / denom
    raise ValueError(f"unknown reduction: {reduction!r}")


def flat_nll(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Mean-token NLL chuẩn (dùng cho Vanilla SFT và để cross-check)."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    shifted_logits = logits[:, :-1].float().reshape(-1, logits.size(-1))
    shifted_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=ignore_index)


def expert_anchor_divergence_loss(
    expert_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    step_spans: Optional[Sequence[Span]] = None,
    gamma_margin: float = 0.05,
    entropy_weighting: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Divergence loss cho một expert so với điểm neo consensus / base (no_grad anchor).
    
    Tính toán từng step để tiết kiệm bộ nhớ tối đa (zero peak VRAM overhead):
    L_div = ReLU(gamma - KL(p_expert || p_anchor))
    """
    if expert_logits.dim() == 3:
        expert_logits = expert_logits[0]
    if anchor_logits.dim() == 3:
        anchor_logits = anchor_logits[0]

    T = expert_logits.shape[0]
    spans = step_spans if step_spans is not None and len(step_spans) > 0 else [(0, T)]

    step_losses: List[torch.Tensor] = []
    step_weights: List[torch.Tensor] = []
    step_kls: List[float] = []

    for start, end in spans:
        start_idx = max(start - 1, 0)
        end_idx = min(end - 1, T)
        if end_idx <= start_idx:
            continue

        step_len = end_idx - start_idx
        chunk_size = 128
        kl_sum = 0.0
        h_sum = 0.0

        for c_start in range(start_idx, end_idx, chunk_size):
            c_end = min(c_start + chunk_size, end_idx)
            e_chunk = expert_logits[c_start:c_end].float()
            a_chunk = anchor_logits[c_start:c_end].float().detach()

            log_p_e = F.log_softmax(e_chunk, dim=-1)
            p_e = log_p_e.exp()
            log_p_a = F.log_softmax(a_chunk, dim=-1)

            kl_toks = (p_e * (log_p_e - log_p_a)).sum(dim=-1).clamp_min(0.0)
            kl_sum = kl_sum + kl_toks.sum()

            if entropy_weighting:
                p_a = log_p_a.exp()
                h_toks = -(p_a * (p_a + eps).log()).sum(dim=-1)
                h_sum = h_sum + h_toks.sum()

        kl_step = kl_sum / max(step_len, 1)
        step_kls.append(float(kl_step.detach()))

        if entropy_weighting:
            h_step = h_sum / max(step_len, 1)
            w_step = torch.tanh(h_step).detach()
        else:
            w_step = expert_logits.new_tensor(1.0)

        if gamma_margin > 0:
            penalty = F.relu(gamma_margin - kl_step)
        else:
            penalty = -kl_step

        step_losses.append(penalty)
        step_weights.append(w_step)

    if not step_losses:
        zero = torch.zeros((), device=expert_logits.device)
        return zero, {"mean_kl": 0.0, "mean_div_loss": 0.0}

    stacked_losses = torch.stack(step_losses)
    stacked_weights = torch.stack(step_weights)
    denom = stacked_weights.sum().clamp_min(1e-8)
    loss = (stacked_weights * stacked_losses).sum() / denom

    mean_kl = sum(step_kls) / max(len(step_kls), 1)
    stats = {
        "mean_kl": mean_kl,
        "mean_div_loss": float(loss.detach()),
    }
    return loss, stats


def controlled_js_divergence_loss(
    logits_per_expert: Sequence[torch.Tensor],
    step_spans: Optional[Sequence[Span]] = None,
    gamma_margin: float = 0.05,
    entropy_weighting: bool = True,
    detach_anchor: bool = True,
    reduction: str = "token",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Controlled Jensen-Shannon Divergence Loss với 4 chốt kiểm soát an toàn:

    1. Detached consensus anchor (detach trên pbar) để tránh vòng lặp gradient.
    2. Entropy soft-weighting: w_t = tanh(H(pbar_t)) chỉ đẩy ở token không chắc chắn.
    3. Margin clamping: L_div = ReLU(gamma - JS) tự ngắt khi đạt đủ độ lệch gamma.
    4. Bounded: 0 <= JS <= log(M).

    Args:
        reduction: ``"token"`` (mặc định) áp margin/weighting riêng từng token
            rồi mới trung bình — mọi token phải tự vượt ``gamma_margin``.
            ``"step"`` gộp JS theo trung bình token trong từng step **trước**
            khi áp margin/weighting — chỉ cần trung bình cả step vượt
            ``gamma_margin``. Khớp đúng granularity với ``V_i`` (cũng là
            trung bình theo step trong ``core/council.py``), nên dễ đẩy được
            V_i qua cổng chặn hơn, đổi lại phân kỳ có thể không đều trong
            step (tập trung ở vài token thay vì lan toả).
    """
    if reduction not in ("token", "step"):
        raise ValueError(f"unknown js reduction: {reduction!r}")

    M = len(logits_per_expert)
    if M < 2:
        zero = torch.zeros((), device=logits_per_expert[0].device)
        return zero, {"mean_js": 0.0, "mean_js_weighted": 0.0}

    squeezed = [lg[0] if lg.dim() == 3 else lg for lg in logits_per_expert]
    T, _ = squeezed[0].shape

    span_lengths: Optional[List[int]] = None
    if step_spans is not None and len(step_spans) > 0:
        token_indices: List[int] = []
        span_lengths = []
        for start, end in step_spans:
            start = max(start, 1)
            end = min(end, T)
            if end > start:
                token_indices.extend(range(start - 1, end - 1))
                span_lengths.append(end - start)
        if not token_indices:
            zero = torch.zeros((), device=squeezed[0].device)
            return zero, {"mean_js": 0.0, "mean_js_weighted": 0.0}
        idx = torch.as_tensor(token_indices, device=squeezed[0].device)
        expert_logits = [lg.index_select(0, idx) for lg in squeezed]
    else:
        expert_logits = squeezed

    num_selected = expert_logits[0].size(0)
    chunk_size = 128
    js_list: List[torch.Tensor] = []
    h_pbar_list: List[torch.Tensor] = []

    for c_start in range(0, num_selected, chunk_size):
        c_end = min(c_start + chunk_size, num_selected)
        c_logits = [lg[c_start:c_end].float() for lg in expert_logits]
        c_probs = [F.softmax(lg, dim=-1) for lg in c_logits]
        c_log_probs = [F.log_softmax(lg, dim=-1) for lg in c_logits]

        c_pbar = torch.stack(c_probs, dim=0).mean(dim=0)
        c_pbar_anchor = c_pbar.detach() if detach_anchor else c_pbar
        c_log_pbar = (c_pbar_anchor + eps).log()

        c_kl_list = []
        for m in range(M):
            c_kl_m = (c_probs[m] * (c_log_probs[m] - c_log_pbar)).sum(dim=-1)
            c_kl_list.append(c_kl_m)

        c_js = torch.stack(c_kl_list, dim=0).mean(dim=0).clamp_min(0.0)
        js_list.append(c_js)

        if entropy_weighting:
            c_h = -(c_pbar.detach() * (c_pbar.detach() + eps).log()).sum(dim=-1)
            h_pbar_list.append(c_h)

    js_per_token = torch.cat(js_list, dim=0)

    if reduction == "step":
        lengths = span_lengths if span_lengths is not None else [js_per_token.numel()]
        js_per_step = torch.stack([chunk.mean() for chunk in js_per_token.split(lengths)])

        if entropy_weighting:
            h_pbar = torch.cat(h_pbar_list, dim=0)
            step_weights = torch.stack(
                [chunk.mean() for chunk in h_pbar.split(lengths)]
            )
            step_weights = torch.tanh(step_weights)
        else:
            step_weights = torch.ones_like(js_per_step)

        if gamma_margin > 0:
            repulsive_penalty = F.relu(gamma_margin - js_per_step)
        else:
            repulsive_penalty = -js_per_step

        denom = step_weights.sum().clamp_min(1e-8)
        loss = (step_weights * repulsive_penalty).sum() / denom

        stats = {
            "mean_js": float(js_per_step.mean().detach()),
            "max_js": float(js_per_step.max().detach()),
            "mean_js_weighted": float(((step_weights * js_per_step).sum() / denom).detach()),
        }
        return loss, stats

    if entropy_weighting:
        h_pbar = torch.cat(h_pbar_list, dim=0)
        token_weights = torch.tanh(h_pbar)
    else:
        token_weights = torch.ones_like(js_per_token)

    if gamma_margin > 0:
        repulsive_penalty = F.relu(gamma_margin - js_per_token)
    else:
        repulsive_penalty = -js_per_token

    denom = token_weights.sum().clamp_min(1e-8)
    loss = (token_weights * repulsive_penalty).sum() / denom

    stats = {
        "mean_js": float(js_per_token.mean().detach()),
        "max_js": float(js_per_token.max().detach()),
        "mean_js_weighted": float(((token_weights * js_per_token).sum() / denom).detach()),
    }
    return loss, stats

