"""SVGD repulsion cho council LoRA experts — thay L_DPP_div bằng gradient dạng đóng.

Ý tưởng (SVGD / GAC-MSO): particle = expert. Không tính loss đa dạng rồi backward;
lực đẩy RBF tính THẲNG (đóng) và cộng vào ``.grad`` ngay trước optimizer.step().

Hai chế độ:
  - ``param``   : kernel trên VECTOR THAM SỐ LoRA. 0 forward thêm. Hook thuần optimizer.
                  θ xa nhau -> K→0 -> repulsion→0: tự tắt, KHÔNG over-diversify (không sập expert).
                  + tuỳ chọn kernel-grad-mix: grad_i ← Σ_j K_ij grad_j / Σ_j K_ij  (kéo expert yếu
                  về hướng gradient của nhóm — chống collapse).
  - ``func``    : lực đẩy RBF trên DỰ ĐOÁN (log-prob) — vẫn là 1 số hạng loss trong vòng lặp
                  per-expert, nhưng bỏ slogdet + Kneedle của DPP. Giữ diversity ở prediction-space
                  (cần cho cổng V).

Update (SVGD, RBF):
    K_ij = exp(-γ ‖z_i − z_j‖²),  γ = 1/(2σ²),  σ² = median(‖z_i−z_j‖²)/(2 ln(M+1))
    φ_repel(z_i) = (1/M) Σ_j 2γ (z_i − z_j) K_ij
    grad_i  ←  [Σ_j K_ij grad_j / Σ_j K_ij]  −  tr_off_k · φ_repel(z_i)
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F


def _adapter_lora_params(model, adapter_name: str):
    """List param LoRA của 1 adapter, thứ tự deterministic (khớp giữa adapter).

    KHÔNG lọc theo requires_grad: sau Pass-2 ``model.set_adapter`` để requires_grad
    của expert không-active = False, nhưng ``.grad`` vẫn còn (backward đã chạy).
    """
    out = []
    for n, p in model.named_parameters():
        if "lora_" in n and (f".{adapter_name}." in n or n.endswith(f".{adapter_name}.weight")):
            out.append(p)
    return out


def _rbf(X: torch.Tensor, sigma: float | None):
    """X (M,D) -> (K (M,M), gamma, sigma). Median heuristic nếu sigma None."""
    M = X.shape[0]
    d2 = torch.cdist(X, X, p=2) ** 2
    off = d2[~torch.eye(M, dtype=torch.bool, device=X.device)]
    if sigma is None:
        med = off.median() if off.numel() else X.new_tensor(1.0)
        h = (med / (2 * math.log(M + 1) + 1e-8)).clamp_min(1e-12)
        s = h.sqrt().item()
    else:
        s = float(sigma)
    gamma = 1.0 / (2 * s * s + 1e-12)
    K = torch.exp(-gamma * d2)
    return K, gamma, s


@torch.no_grad()
def apply_svgd_param(model, adapter_names: List[str], tr_off_k: float = 1.0,
                     sigma: float | None = None, grad_mix: bool = True) -> Dict[str, float]:
    """Cộng repulsion SVGD (param-space) vào .grad của từng adapter. Gọi TRƯỚC clip_and_step.

    Trả stats để log. Bỏ qua nếu adapter nào chưa có grad.
    """
    plists = [_adapter_lora_params(model, a) for a in adapter_names]
    M = len(plists)
    if M < 2 or any(len(pl) == 0 for pl in plists):
        return {}
    if any(p.grad is None for pl in plists for p in pl):
        return {}
    sizes = [p.numel() for p in plists[0]]
    X = torch.stack([torch.cat([p.detach().reshape(-1) for p in pl]) for pl in plists])  # (M,D)
    G = torch.stack([torch.cat([p.grad.reshape(-1) for p in pl]) for pl in plists])       # (M,D)

    K, gamma, s = _rbf(X, sigma)
    Ksum = K.sum(dim=1, keepdim=True).clamp_min(1e-8)                                      # (M,1)

    # 1) kernel-grad-mix (trên grad TASK, trước repulsion) — trung bình có trọng số, giữ magnitude
    G_new = (K @ G) / Ksum if grad_mix else G

    # 2) repulsion: φ_i = (2γ/M)(X_i·Ksum_i − Σ_j K_ij X_j)
    repel = (2.0 * gamma / M) * (X * Ksum - K @ X)                                         # (M,D)
    G_new = G_new - tr_off_k * repel

    # ghi lại vào .grad
    for m, pl in enumerate(plists):
        offset = 0
        row = G_new[m]
        for p, npar in zip(pl, sizes):
            p.grad.copy_(row[offset:offset + npar].reshape(p.shape))
            offset += npar

    off = K[~torch.eye(M, dtype=torch.bool, device=K.device)]
    return {"svgd_K_offdiag": float(off.mean()), "svgd_sigma": s,
            "svgd_repel_norm": float(repel.norm(dim=1).mean()),
            "svgd_dist": float((torch.cdist(X, X).sum() / (M * (M - 1))).clamp_min(0))}


def svgd_func_repulsion(logp_list: List[torch.Tensor], m: int, tr_off_k: float = 1.0,
                        sigma: float | None = None):
    """Lực đẩy RBF trên dự đoán. logp_list[k] = (P, V) log-prob của expert k trên cùng vị trí;
    CHỈ logp_list[m] có grad, còn lại detach. Trả (loss_repel_m, stats).

    loss = tr_off_k · (1/M) Σ_{j≠m} K_mj    (giảm loss = giảm K = đẩy xa)
    """
    M = len(logp_list)
    zi = logp_list[m].reshape(-1)
    d2 = []
    for j in range(M):
        if j == m:
            continue
        d2.append(((zi - logp_list[j].reshape(-1).detach()) ** 2).sum())
    d2 = torch.stack(d2)                                   # (M-1,)
    if sigma is None:
        s = (d2.detach().median() / (2 * math.log(M + 1) + 1e-8)).clamp_min(1e-12).sqrt()
    else:
        s = torch.as_tensor(float(sigma), device=zi.device)
    gamma = 1.0 / (2 * s * s + 1e-12)
    K = torch.exp(-gamma * d2)                             # (M-1,)
    loss = tr_off_k * K.mean()
    return loss, {"svgd_func_K": float(K.detach().mean()), "svgd_func_sigma": float(s)}
