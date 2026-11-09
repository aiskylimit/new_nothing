"""Step-level Gradient Dropout (proposal eq. 7-8, spec §7.1).

Với mỗi cặp (step i, expert m):

    d_{i,m} ~ Bernoulli(1 - p_drop)
    gtilde_{i,m} = d_{i,m} * grad_{phi_m} L_i^{(m)}

Triển khai bằng **custom autograd function**: giá trị forward giữ nguyên (log
loss vẫn là loss thật), chỉ gradient bị scale. Cố ý KHÔNG zero cứng loss vì:
  * loss report sẽ sai lệch theo p_drop,
  * nếu cả một step bị zero ở mọi expert thì loss = 0 và optimizer vẫn bước.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class _ScaleGradient(torch.autograd.Function):
    """Identity ở forward, nhân gradient với ``scale`` ở backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor):  # type: ignore[override]
        ctx.save_for_backward(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (scale,) = ctx.saved_tensors
        return grad_out * scale.to(grad_out.dtype), None


def scale_gradient(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên giá trị ``x``, scale gradient chảy ngược qua ``x``."""
    return _ScaleGradient.apply(x, scale)


def sample_dropout_mask(
    shape: Tuple[int, ...],
    p_drop: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """d ~ Bernoulli(1 - p_drop), shape tuỳ ý. ``p_drop=0`` -> toàn 1."""
    if not 0.0 <= p_drop < 1.0:
        raise ValueError(f"p_drop phải nằm trong [0, 1), nhận {p_drop}")
    if p_drop == 0.0:
        return torch.ones(shape, device=device)
    probs = torch.full(shape, 1.0 - p_drop, device=device)
    return torch.bernoulli(probs, generator=generator)


def apply_step_gradient_dropout(
    step_losses: torch.Tensor,
    p_drop: float,
    generator: Optional[torch.Generator] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Áp gradient dropout lên ma trận loss ``(M, N)``.

    Args:
        step_losses: ``L[m, i]`` — NLL của expert m trên step i (có grad).
        p_drop: xác suất drop gradient của một cặp (step, expert).
        generator: RNG để tái lập.
        mask: dùng mask cho sẵn thay vì sample (cho unit test).

    Returns:
        ``(losses_with_scaled_grad, mask)``. Giá trị của phần tử [m, i] bằng
        đúng ``step_losses[m, i]``; gradient của nó bị nhân ``mask[m, i]``.
    """
    if mask is None:
        mask = sample_dropout_mask(
            tuple(step_losses.shape), p_drop, device=step_losses.device, generator=generator
        )
    mask = mask.to(step_losses.device, step_losses.dtype)
    return scale_gradient(step_losses, mask), mask
