"""Gate router theo step (proposal eq. 10, spec §7.3).

    g_i = softmax(W_gate h_i) ∈ R^M

``h_i`` là hidden state của **token cuối step i** lấy từ base model (đã tắt mọi
adapter). Gate là tham số DUY NHẤT được train trong Phase 2.

Module này dùng chung cho train (``train/phase2_gate.py``) và inference
(``eval/gate_inference.py``) để đảm bảo cùng một định nghĩa h_i.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

Span = Tuple[int, int]


class StepGate(nn.Module):
    """Linear router hidden_size -> M, không bias (đúng eq. 10)."""

    def __init__(self, hidden_size: int, n_experts: int, temperature: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.temperature = temperature
        self.proj = nn.Linear(hidden_size, n_experts, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden: (N, H)`` -> ``g: (N, M)`` đã softmax."""
        logits = self.proj(hidden.to(self.proj.weight.dtype)) / self.temperature
        return torch.softmax(logits, dim=-1)

    def save(self, out_dir: str, extra: Optional[dict] = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "gate.pt")
        torch.save(self.state_dict(), path)
        meta = {
            "hidden_size": self.hidden_size,
            "n_experts": self.n_experts,
            "temperature": self.temperature,
        }
        meta.update(extra or {})
        with open(os.path.join(out_dir, "gate_config.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return path

    @classmethod
    def load(cls, out_dir: str, map_location="cpu") -> "StepGate":
        with open(os.path.join(out_dir, "gate_config.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        gate = cls(meta["hidden_size"], meta["n_experts"], meta.get("temperature", 1.0))
        gate.load_state_dict(torch.load(os.path.join(out_dir, "gate.pt"), map_location=map_location))
        return gate


def step_hidden_states(
    hidden: torch.Tensor, spans: Sequence[Span]
) -> torch.Tensor:
    """Lấy hidden state của token cuối mỗi step.

    Args:
        hidden: ``(T, H)`` hoặc ``(1, T, H)`` — hidden state lớp cuối của base.
        spans: span token ``[start, end)`` trong toạ độ ``input_ids``.

    Returns:
        ``(N, H)``.
    """
    if hidden.dim() == 3:
        hidden = hidden[0]
    seq_len = hidden.shape[0]
    rows = []
    for start, end in spans:
        index = min(max(end - 1, 0), seq_len - 1)
        rows.append(hidden[index])
    return torch.stack(rows, dim=0)


def load_balancing_loss(g: torch.Tensor) -> torch.Tensor:
    """Auxiliary load-balancing (kiểu Switch Transformer), tuỳ chọn.

    ``g: (N, M)``. Trả về ``M * sum_m f_m * P_m`` với ``f_m`` là tỉ lệ step
    route (argmax) về expert m và ``P_m`` là xác suất trung bình. Giá trị tối
    thiểu = 1 khi cân bằng hoàn hảo.

    KHÔNG có trong proposal — mặc định ``lambda_balance = 0``. Chỉ bật khi gate
    bị collapse về một expert (xem README §Rủi ro).
    """
    n_experts = g.shape[-1]
    mean_prob = g.mean(dim=0)
    hard = torch.zeros_like(g).scatter_(1, g.argmax(dim=-1, keepdim=True), 1.0)
    frac = hard.mean(dim=0)
    return n_experts * (frac * mean_prob).sum()
