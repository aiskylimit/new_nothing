"""§4.3 — Phân hóa đầu ra: Dynamic Top-k (Kneedle) + DPP log-det.

Eq (12)-(16) của proposal:

  - Với mỗi token, lấy phân phối đồng thuận p̄ = mean_m p^(m) (Eq 8).
  - Chuẩn hóa rank/giá trị về [0,1] (Eq 12), tìm điểm khuỷu k (Eq 13) — token
    có (1 - x_j) - y_j lớn nhất → cắt lấy V_k token đầu.
  - Bỏ ground-truth y* khỏi V_k → tập ứng viên C (Eq 14).
  - Mỗi expert m: v^(m) = [p^(m)(c)]_{c in C}, chuẩn hóa L2 → ṽ^(m).
  - Gram G_{p,q} = <ṽ^(p), ṽ^(q)>; l_DPP = -log det(G + αI) (Eq 15).
  - L_DPP_div = trung bình l_DPP theo step rồi theo toàn chuỗi (Eq 16).

Ghi chú thực thi:
  - Hàm nhận **log-prob đầy đủ** của từng expert đã được index về đúng các vị
    trí token dự đoán (tọa độ giống core/losses.token_logprobs: logits[t] dự
    đoán input_ids[t+1]). Caller lo phần forward + shift.
  - Gradient chảy qua ṽ^(m) (do đó qua log-prob expert). p̄, k, C đều không cần
    grad (detach) — giống "detached consensus anchor" của nhánh JS.
  - Chunk theo token để không giữ (N_tok, V) fp32 một lúc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Span = Tuple[int, int]


def kneedle_topk(pbar_sorted_desc: torch.Tensor) -> int:
    """Eq (12)-(13). ``pbar_sorted_desc`` : (V,) xác suất đã sắp giảm dần.

    Trả về k = số token "mang thông tin" (>=1). Chuẩn hóa min-max cả trục rank
    lẫn trục giá trị về [0,1] rồi lấy argmax của (1 - x_j) - y_j.
    """
    v = pbar_sorted_desc.detach().float()
    n = v.numel()
    if n <= 1:
        return int(n)
    p_max = v[0]
    p_min = v[-1]
    denom = (p_max - p_min).clamp_min(1e-12)
    y = (v - p_min) / denom                       # (V,) in [0,1], giảm dần
    x = torch.arange(1, n + 1, device=v.device, dtype=v.dtype) / n
    score = (1.0 - x) - y
    k = int(torch.argmax(score).item()) + 1
    return max(k, 1)


def _token_dpp_logdet(
    logp_experts_tok: Sequence[torch.Tensor],   # M tensors (V,), có grad
    gold_id: Optional[int],
    alpha: float,
    max_cand: int,
    eps: float,
) -> Optional[torch.Tensor]:
    """l_DPP cho MỘT token (Eq 14-15). None nếu |C| <= 1."""
    M = len(logp_experts_tok)
    with torch.no_grad():
        pbar = torch.stack([lp.detach() for lp in logp_experts_tok], dim=0).exp().mean(0)  # (V,)
        sorted_p, sorted_idx = torch.sort(pbar, descending=True)
        k = kneedle_topk(sorted_p)
        k = min(k, max_cand)
        cand = sorted_idx[:k]
        if gold_id is not None:
            cand = cand[cand != gold_id]
        if cand.numel() <= 1:
            return None

    # vector đặc trưng có grad: p^(m) trên tập ứng viên, chuẩn hóa L2
    vs = []
    for lp in logp_experts_tok:
        v = lp.index_select(0, cand).exp()                 # (|C|,)
        v = v / (v.norm() + eps)
        vs.append(v)
    Vmat = torch.stack(vs, dim=0)                          # (M, |C|)
    G = Vmat @ Vmat.t()                                    # (M, M)
    G = G + alpha * torch.eye(M, device=G.device, dtype=G.dtype)
    # -log det(G): dùng slogdet cho ổn định số. G = ṼṼ^T + αI đối xứng nửa xác
    # định dương nên sign luôn +1; vẫn kiểm tra phòng lỗi số.
    sign, logabsdet = torch.linalg.slogdet(G)
    if float(sign.detach()) <= 0:
        return None
    return -logabsdet


@dataclass
class DppStats:
    mean_logdet_loss: float
    mean_cand: float
    frac_valid_tokens: float
    n_steps: int


def dpp_diversity_loss(
    logp_per_expert: Sequence[torch.Tensor],   # M tensors (N_tok, V) — log-softmax, có grad
    gold_ids: torch.Tensor,                    # (N_tok,) long — token đúng tại mỗi vị trí
    step_lengths: Sequence[int],               # tổng = N_tok, phân N_tok về S step
    alpha: float = 1e-5,
    max_cand: int = 64,
    token_chunk: int = 64,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, DppStats]:
    """Eq (16). Trả (loss scalar có grad, thống kê).

    loss = mean_step( mean_token_in_step( l_DPP ) ). Token có |C|<=1 bị bỏ; step
    không còn token hợp lệ → l_DPP step = 0 (không kéo loss).
    """
    M = len(logp_per_expert)
    device = logp_per_expert[0].device
    if M < 2:
        return torch.zeros((), device=device), DppStats(0.0, 0.0, 0.0, 0)

    N_tok = logp_per_expert[0].shape[0]
    per_token_loss: List[Optional[torch.Tensor]] = [None] * N_tok
    cand_sizes: List[int] = []

    for c0 in range(0, N_tok, token_chunk):
        c1 = min(c0 + token_chunk, N_tok)
        for t in range(c0, c1):
            tok_logps = [lp[t] for lp in logp_per_expert]
            gold = int(gold_ids[t].item())
            l = _token_dpp_logdet(tok_logps, gold, alpha, max_cand, eps)
            if l is not None:
                per_token_loss[t] = l

    # gộp theo step
    step_losses: List[torch.Tensor] = []
    valid_tok = 0
    offset = 0
    for L_i in step_lengths:
        chunk = [per_token_loss[offset + j] for j in range(L_i)]
        offset += L_i
        chunk = [c for c in chunk if c is not None]
        if not chunk:
            continue
        valid_tok += len(chunk)
        step_losses.append(torch.stack(chunk).mean())

    if not step_losses:
        return torch.zeros((), device=device, requires_grad=True) * logp_per_expert[0].sum() * 0.0, \
            DppStats(0.0, 0.0, 0.0, 0)

    loss = torch.stack(step_losses).mean()
    stats = DppStats(
        mean_logdet_loss=float(loss.detach()),
        mean_cand=(sum(cand_sizes) / len(cand_sizes)) if cand_sizes else 0.0,
        frac_valid_tokens=valid_tok / max(N_tok, 1),
        n_steps=len(step_losses),
    )
    return loss, stats
