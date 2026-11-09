"""Entropy-Guided Temperature Selection (EGTS) — vendored từ CurioSFT.

Nguồn: https://github.com/HaoooWang/CurioSFT
  custom_verl/recipe/curio_sft/dp_actor.py :: compute_tau_from_logits (commit a83b129)
Bài báo: "Learning While Staying Curious: Entropy-Preserving SFT via Adaptive
  Self-Distillation for Large Reasoning Models" (ACL 2026), arXiv:2602.02244, App. D + Fig. 6.
License gốc: Apache-2.0 (xem vendor/curiosft/README.md).

ĐÃ CHỈNH SO VỚI BẢN GỐC:
  - Bỏ phụ thuộc verl / flash-attn: hàm thuần torch, nhận logits ``(N, V)``.
  - Tách hằng số ra tham số keyword (h_pivot, gamma, delta_max) — mặc định giữ
    NGUYÊN giá trị repo gốc (c=1.2, scale=2.0, 0.5, tau∈[1.1,1.5], topk=512, iters=20).
  - Thêm docstring + type hint; logic binary search giữ nguyên từng dòng.

Quan hệ với `core/`:
  - `core/council.py::entropy_from_logits` tính H(softmax(logits)) đầy đủ vocab;
    hàm này xấp xỉ H bằng top-k logits (App. D) để chạy nhanh trong training loop.
  - EGTS chỉ tạo ra `tau_i` cho TEACHER (base đóng băng). Student vẫn ở tau=1.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


@torch.no_grad()
def compute_tau_from_logits(
    logits: torch.Tensor,
    *,
    tau_min: float = 1.1,
    tau_max: float = 1.5,
    topk: int = 512,
    iters: int = 20,
    h_pivot: float = 1.2,
    gamma: float = 2.0,
    delta_max: float = 0.5,
    low_entropy_cutoff: float = 0.03,
    apply_low_entropy_cutoff: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chọn nhiệt độ teacher theo từng token bằng binary search (EGTS, eq. 7 + App. D).

    Với mỗi token: ``H_now = H(softmax(z))``, entropy mục tiêu
    ``H* = H_now + delta_max * sigmoid(gamma * (H_now - h_pivot))``, rồi tìm
    ``tau ∈ [tau_min, tau_max]`` sao cho ``H(softmax(z / tau)) ≈ H*``.
    Entropy được xấp xỉ trên ``topk`` logits cao nhất.

    Args:
        logits: ``(N, V)`` — logits teacher (base đóng băng) tại các vị trí token.
        tau_min, tau_max: khoảng clip nhiệt độ.
        topk: số logits dùng để xấp xỉ entropy (App. D, k=512).
        iters: số vòng binary search.
        h_pivot, gamma, delta_max: hằng số eq. (7). Mặc định = repo gốc.
        low_entropy_cutoff: token có ``H_now <= cutoff`` giữ nguyên ``tau_min``
            (không đẩy entropy ở token gần như xác định — factual token).
        apply_low_entropy_cutoff: nếu ``True`` áp cutoff bằng ``H_now`` (top-k, xấp xỉ).
            Repo gốc (``dp_actor.py:205``) áp cutoff ở caller bằng entropy **full-vocab**;
            ``core/sed.py`` truyền ``False`` rồi tự áp bằng full entropy cho khớp repo.

    Returns:
        ``(H_now, tau)`` — mỗi cái shape ``(N,)``. ``H_now`` là entropy **top-k** (App. D).
    """
    assert logits.dim() == 2, "logits phải là (N, V)"
    n, vocab = logits.shape
    device, dtype = logits.device, logits.dtype
    k = min(topk, vocab)
    z = logits

    def entropy_topk(z_: torch.Tensor, tau_: torch.Tensor) -> torch.Tensor:
        if k >= vocab:
            logp = torch.log_softmax(z_ / tau_, dim=-1)
            return -(logp.exp() * logp).sum(dim=-1)
        vals, _ = torch.topk(z_, k=k, dim=-1)
        logp_k = torch.log_softmax(vals / tau_, dim=-1)
        return -(logp_k.exp() * logp_k).sum(dim=-1)

    ones = torch.ones(n, 1, device=device, dtype=dtype)
    h_now = entropy_topk(z, ones)  # (N,)

    delta = torch.sigmoid((h_now - h_pivot) * gamma) * delta_max
    h_star = (h_now + delta).unsqueeze(-1)

    tau_lo = torch.full((n, 1), tau_min, device=device, dtype=dtype)
    tau_hi = torch.full((n, 1), tau_max, device=device, dtype=dtype)
    h_lo = entropy_topk(z, tau_lo).unsqueeze(-1)
    h_hi = entropy_topk(z, tau_hi).unsqueeze(-1)

    tau = torch.where(
        h_star <= h_lo, tau_lo,
        torch.where(h_star >= h_hi, tau_hi, (tau_lo + tau_hi) / 2),
    )

    for _ in range(iters):
        h_mid = entropy_topk(z, tau).unsqueeze(-1)
        go_hi = h_mid < h_star  # entropy còn thấp -> tăng tau
        tau_lo = torch.where(go_hi, tau, tau_lo)
        tau_hi = torch.where(go_hi, tau_hi, tau)
        tau = (tau_lo + tau_hi) / 2

    tau = tau.squeeze(-1)
    if apply_low_entropy_cutoff:
        tau = torch.where(h_now <= low_entropy_cutoff, tau.new_full((), tau_min), tau)
    return h_now, tau


@torch.no_grad()
def teacher_logprobs_with_egts(
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    adaptive: bool = True,
    fixed_tau: float = 1.3,
    **egts_kwargs,
) -> torch.Tensor:
    """log p_teacher(target_ids) sau khi scale nhiệt độ (per-token nếu ``adaptive``).

    Args:
        teacher_logits: ``(T, V)`` logits base đóng băng (đã align: hàng t dự đoán target_ids[t]).
        target_ids: ``(T,)`` id token teacher (expert token).
        adaptive: True -> EGTS per-token; False -> dùng ``fixed_tau`` cho mọi token
            (ứng với ablation "w/o Adaptive Temp." của bài, tau=1.3).

    Returns:
        ``(T,)`` log-prob của expert token dưới teacher đã scale.
    """
    if adaptive:
        _, tau = compute_tau_from_logits(teacher_logits, **egts_kwargs)
        scaled = teacher_logits / tau.unsqueeze(-1)
    else:
        scaled = teacher_logits / fixed_tau
    logp = torch.log_softmax(scaled.float(), dim=-1)
    return logp.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
