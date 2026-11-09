"""Step weighting: từ (U, V) hoặc entropy -> w_i (spec §4.4, proposal eq. 16).

    Uhat, Vhat = zscore(U), zscore(V)
    w_i = 1 + lambda_U * tanh(Uhat_i) + lambda_D * tanh(Vhat_i)

``tanh`` chặn w_i trong ``[1 - lU - lD, 1 + lU + lD]`` nên với mặc định
0.5/0.5 thì w_i ∈ [0, 2]; ``clip_min`` giữ w_i > 0 để loss không bị đảo dấu.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score trên chiều cuối. Std ~ 0 (mọi step giống nhau) -> trả về 0."""
    x = x.float()
    mean = x.mean()
    std = x.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(x)
    return (x - mean) / std


def zscore_with_stats(
    x: torch.Tensor, mean: float, std: float, eps: float = 1e-6
) -> torch.Tensor:
    """Z-score dùng thống kê cho sẵn (scope = batch/corpus)."""
    if std < eps:
        return torch.zeros_like(x.float())
    return (x.float() - mean) / std


def step_weights(
    U: torch.Tensor,
    V: torch.Tensor,
    lambda_U: float = 0.5,
    lambda_D: float = 0.5,
    *,
    stats: Optional[dict] = None,
    clip_min: float = 1e-3,
    disagreement_mode: str = "raw",
    eps: float = 1e-8,
) -> torch.Tensor:
    """w_i từ hội đồng (eq. 16).

    Args:
        U, V: ``(N,)``.
        lambda_U, lambda_D: hệ số trong eq. (16). Đặt một trong hai = 0 để chạy
            ablation ``U-only`` / ``V-only``; cả hai = 0 -> w_i ≡ 1 (``uniform``).
        stats: nếu truyền ``{"U_mean","U_std","V_mean","V_std"}`` thì z-score
            theo thống kê corpus (``--zscore_scope corpus``); mặc định z-score
            trong nội bộ một mẫu.
        disagreement_mode:
            ``"raw"``        - dùng V nguyên bản. ĐÚNG proposal, là mặc định.
            ``"normalized"`` - dùng ``rho = V / U``, tức bất đồng TƯƠNG ĐỐI.

    Lý do có ``"normalized"``: MI bị chặn bởi entropy có sẵn, nên ``V ~ U * rho``.
    Đo trên 14.137 step / 2 council độc lập (p_drop 0.1 vs 0.9):
    Spearman(U, V) = 0.91 (V gần như là U viết lại), trong khi Spearman(U, rho)
    = 0.09 và rho tái lập chéo giữa 2 council ở Spearman 0.32. Chia cho U tách
    được "thực sự bất đồng bao nhiêu" khỏi "có bao nhiêu chỗ để bất đồng".
    """
    if disagreement_mode == "raw":
        D = V
    elif disagreement_mode == "normalized":
        D = V / torch.clamp(U.float(), min=eps)
    else:
        raise ValueError(f"unknown disagreement_mode: {disagreement_mode!r}")

    if stats is None:
        Uhat, Dhat = zscore(U), zscore(D)
    else:
        Uhat = zscore_with_stats(U, stats["U_mean"], stats["U_std"])
        Dhat = zscore_with_stats(D, stats["V_mean"], stats["V_std"])
    w = 1.0 + lambda_U * torch.tanh(Uhat) + lambda_D * torch.tanh(Dhat)
    return w.clamp_min(clip_min)


def entropy_weights(
    H: torch.Tensor,
    lambda_H: float = 0.5,
    *,
    stats: Optional[dict] = None,
    clip_min: float = 1e-3,
) -> torch.Tensor:
    """w_i^{ent} = 1 + lambda_H * tanh(zscore(H)) — baseline Entropy Selection."""
    if stats is None:
        Hhat = zscore(H)
    else:
        Hhat = zscore_with_stats(H, stats["H_mean"], stats["H_std"])
    return (1.0 + lambda_H * torch.tanh(Hhat)).clamp_min(clip_min)


def prune_mask_from_entropy(H: torch.Tensor, kappa: float) -> torch.Tensor:
    """Mask giữ lại top-(1-kappa) step theo entropy (mode ``prune`` của §6).

    Trả về bool ``(N,)``: True = giữ, False = step bị thay bằng ``[SKIP]``.
    """
    n = H.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    n_drop = int(round(kappa * n))
    keep = torch.ones(n, dtype=torch.bool)
    if n_drop > 0:
        lowest = torch.topk(H.float(), k=min(n_drop, n), largest=False).indices
        keep[lowest] = False
    return keep


def corpus_stats(values: Sequence[torch.Tensor]) -> dict:
    """Mean/std gộp qua nhiều mẫu — cho ``--zscore_scope corpus``."""
    flat = torch.cat([v.reshape(-1).float() for v in values if v.numel() > 0])
    return {"mean": float(flat.mean()), "std": float(flat.std(unbiased=False))}
