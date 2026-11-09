"""L_rep — regularizer ép M LoRA expert khác nhau (proposal eq. 9, spec §7.1).

Đo tương đồng giữa hai expert bằng cosine similarity của **delta-weight**
``dW = scaling * B @ A`` trên từng target module, rồi lấy trung bình qua mọi
module và mọi cặp expert.

Không bao giờ dựng ``dW`` (out x in, tới hàng chục triệu phần tử). Dùng đẳng
thức chỉ cần ma trận r x r:

    <B_p A_p, B_q A_q>_F = sum( (B_p^T B_q) ⊙ (A_p A_q^T) )

với ``B: (out, r)``, ``A: (r, in)``. Chi phí O(r^2 * (out + in)) thay vì
O(out * in).

GHI CHÚ VỀ DẤU (spec §7.1 viết "âm của mean pairwise cosine"): tổng loss là
``L = sum NLL + lambda_rep * L_rep`` với ``lambda_rep > 0``, nên để *phạt* sự
giống nhau thì ``L_rep`` phải **tăng theo** độ tương đồng. Mặc định dùng
``cos^2`` (phạt cả tương quan âm lẫn dương, tối ưu tại trực giao). ``metric=
"neg_cos"`` tái lập đúng chữ trong spec và được giữ lại cho ablation, nhưng nó
kéo các expert về phía đối xứng nhau (cos -> -1), không phải trực giao.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_EPS = 1e-8


def _iter_lora_modules(model, adapter_names: Sequence[str]):
    """Yield ``(module_name, {adapter: (A, B, scaling)})`` cho mọi LoRA layer."""
    for name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue
        if not all(a in lora_A and a in lora_B for a in adapter_names):
            continue
        entry: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}
        scalings = getattr(module, "scaling", {})
        for adapter in adapter_names:
            A = lora_A[adapter].weight
            B = lora_B[adapter].weight
            scaling = float(scalings.get(adapter, 1.0)) if isinstance(scalings, dict) else 1.0
            entry[adapter] = (A, B, scaling)
        yield name, entry


def pairwise_delta_cosine(
    A_p: torch.Tensor,
    B_p: torch.Tensor,
    s_p: float,
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    s_q: float,
) -> torch.Tensor:
    """cos(dW_p, dW_q) qua đẳng thức r x r; trả scalar tensor có grad."""
    A_p, B_p = A_p.float(), B_p.float()
    A_q, B_q = A_q.float(), B_q.float()

    def frob_inner(Bx, Ax, By, Ay):
        return ((Bx.T @ By) * (Ax @ Ay.T)).sum()

    inner = s_p * s_q * frob_inner(B_p, A_p, B_q, A_q)
    norm_p = (s_p**2 * frob_inner(B_p, A_p, B_p, A_p)).clamp_min(_EPS).sqrt()
    norm_q = (s_q**2 * frob_inner(B_q, A_q, B_q, A_q)).clamp_min(_EPS).sqrt()
    return inner / (norm_p * norm_q + _EPS)


def pairwise_dual_orthogonal(
    A_p: torch.Tensor,
    B_p: torch.Tensor,
    A_q: torch.Tensor,
    B_q: torch.Tensor,
) -> torch.Tensor:
    """Dual-Orthogonal Subspace penalty (OMoE / D2MoRA).
    
    Ép không gian con LoRA của A (input projection) và B (output projection)
    trực giao độc lập:
        L_ortho = ||A_p A_q^T||_F^2 / (||A_p||_F^2 ||A_q||_F^2) + ||B_p^T B_q||_F^2 / (||B_p||_F^2 ||B_q||_F^2)
    """
    A_p, B_p = A_p.float(), B_p.float()
    A_q, B_q = A_q.float(), B_q.float()

    # Gram matrices trong không gian nhỏ r x r
    G_A = A_p @ A_q.T  # (r, r)
    G_B = B_p.T @ B_q  # (r, r)

    norm_A_p = (A_p ** 2).sum().clamp_min(_EPS)
    norm_A_q = (A_q ** 2).sum().clamp_min(_EPS)
    norm_B_p = (B_p ** 2).sum().clamp_min(_EPS)
    norm_B_q = (B_q ** 2).sum().clamp_min(_EPS)

    loss_A = (G_A ** 2).sum() / (norm_A_p * norm_A_q)
    loss_B = (G_B ** 2).sum() / (norm_B_p * norm_B_q)
    return loss_A + loss_B


def lora_diversity_loss(
    model,
    adapter_names: Sequence[str],
    metric: str = "cos2",
    module_filter: Optional[Iterable[str]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_rep + thống kê để log.

    Args:
        model: PeftModel mang đủ M adapter.
        adapter_names: tên M expert.
        metric: ``cos2`` (mặc định) | ``abs_cos`` | ``neg_cos`` | ``dual_ortho``.
        module_filter: nếu truyền, chỉ tính trên module có tên chứa một trong
            các chuỗi này (ví dụ ``["q_proj", "down_proj"]`` để tiết kiệm).

    Returns:
        ``(L_rep, {"mean_pairwise_cosine": ..., "n_modules": ...})``.
        Không có cặp nào (M = 1) -> loss 0.
    """
    if len(adapter_names) < 2:
        zero = torch.zeros((), device=next(model.parameters()).device)
        return zero, {"mean_pairwise_cosine": 0.0, "n_modules": 0.0, "n_pairs": 0.0}

    cosines: List[torch.Tensor] = []
    ortho_losses: List[torch.Tensor] = []
    n_modules = 0
    for module_name, entry in _iter_lora_modules(model, adapter_names):
        if module_filter is not None and not any(f in module_name for f in module_filter):
            continue
        n_modules += 1
        for p in range(len(adapter_names)):
            for q in range(p + 1, len(adapter_names)):
                A_p, B_p, s_p = entry[adapter_names[p]]
                A_q, B_q, s_q = entry[adapter_names[q]]
                cosines.append(pairwise_delta_cosine(A_p, B_p, s_p, A_q, B_q, s_q))
                if metric == "dual_ortho":
                    ortho_losses.append(pairwise_dual_orthogonal(A_p, B_p, A_q, B_q))

    if not cosines:
        zero = torch.zeros((), device=next(model.parameters()).device)
        return zero, {"mean_pairwise_cosine": 0.0, "n_modules": 0.0, "n_pairs": 0.0}

    cos = torch.stack(cosines)
    if metric == "cos2":
        loss = (cos**2).mean()
    elif metric == "abs_cos":
        loss = cos.abs().mean()
    elif metric == "neg_cos":
        loss = -cos.mean()
    elif metric == "dual_ortho":
        loss = torch.stack(ortho_losses).mean()
    else:
        raise ValueError(f"unknown rep metric: {metric!r}")

    stats = {
        "mean_pairwise_cosine": float(cos.mean().detach()),
        "max_pairwise_cosine": float(cos.max().detach()),
        "n_modules": float(n_modules),
        "n_pairs": float(cos.numel()),
    }
    if metric == "dual_ortho":
        stats["mean_ortho_loss"] = float(loss.detach())
    return loss, stats


@torch.no_grad()
def report_pairwise_similarity(model, adapter_names: Sequence[str]) -> Dict[str, float]:
    """Ma trận cosine trung bình giữa từng cặp expert — dùng cho §13 bước 5."""
    sums: Dict[Tuple[int, int], float] = {}
    count = 0
    for _, entry in _iter_lora_modules(model, adapter_names):
        count += 1
        for p in range(len(adapter_names)):
            for q in range(p + 1, len(adapter_names)):
                A_p, B_p, s_p = entry[adapter_names[p]]
                A_q, B_q, s_q = entry[adapter_names[q]]
                value = float(pairwise_delta_cosine(A_p, B_p, s_p, A_q, B_q, s_q))
                sums[(p, q)] = sums.get((p, q), 0.0) + value
    if count == 0:
        return {}
    return {f"cos(expert_{p},expert_{q})": total / count for (p, q), total in sums.items()}
