"""Tín hiệu hội đồng expert: U_i (mean uncertainty) và V_i (disagreement).

Công thức (proposal §5, spec §4.3), entropy tính bằng **nats**:

    E_i^{(m)} = (1/T_i) * sum_t H(phat[m, i, t])
    U_i       = (1/M) * sum_m E_i^{(m)}
    pbar_{i,t}= (1/M) * sum_m phat[m, i, t]
    V_i       = (1/T_i) * sum_t [ H(pbar_{i,t}) - (1/M) sum_m H(phat[m,i,t]) ]

V_i chính là mutual information I(Y; M | s_i) nên về lý thuyết >= 0; do sai số
float nó có thể âm rất nhẹ → luôn ``clamp(V_i, min=0)``.

RÀNG BUỘC BỘ NHỚ (spec §14): **không** materialize tensor ``(M, N, T, V)``.
Driver ở đây stream theo chunk token, mỗi lúc chỉ giữ ``(C, V)`` cho expert hiện
tại cộng một accumulator ``(C, V)`` cho ``sum_m phat`` — độc lập với M và T.

Quy ước index (giống ``core/losses.py``): mảng entropy trả về được đánh index
theo **vị trí token được dự đoán**, tức ``H[t]`` là entropy của phân phối dự
đoán ``input_ids[t]`` (= entropy của ``logits[t - 1]``). ``H[0]`` không hợp lệ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

Span = Tuple[int, int]

_LOG_EPS = 1e-12


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """H(softmax(logits)) theo nats. ``logits: (..., V)`` -> ``(...)``."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """H(p) theo nats cho phân phối đã chuẩn hoá."""
    p = probs.float()
    return -(p * torch.log(p.clamp_min(_LOG_EPS))).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Accumulator: gom entropy theo token, không giữ vocab dist
# --------------------------------------------------------------------------- #
class CouncilAccumulator:
    """Gom ``H_m[t]`` và ``H(pbar)[t]`` theo từng chunk token.

    Index đầu vào (``target_start``) là vị trí token **được dự đoán**.
    """

    def __init__(self, n_experts: int, seq_len: int, device=None):
        self.n_experts = n_experts
        self.seq_len = seq_len
        dev = device or torch.device("cpu")
        self.h_experts = torch.zeros(n_experts, seq_len, dtype=torch.float32, device=dev)
        self.h_mean_dist = torch.zeros(seq_len, dtype=torch.float32, device=dev)
        self.valid = torch.zeros(seq_len, dtype=torch.bool, device=dev)

    def add_chunk(
        self,
        target_start: int,
        h_experts_chunk: torch.Tensor,  # (M, C)
        h_pbar_chunk: torch.Tensor,  # (C,)
    ) -> None:
        length = h_pbar_chunk.numel()
        end = min(target_start + length, self.seq_len)
        if end <= target_start:
            return
        keep = end - target_start
        dev = self.h_experts.device
        self.h_experts[:, target_start:end] = h_experts_chunk[:, :keep].to(dev, torch.float32)
        self.h_mean_dist[target_start:end] = h_pbar_chunk[:keep].to(dev, torch.float32)
        self.valid[target_start:end] = True

    def signals(self, step_spans: Sequence[Span]) -> "CouncilSignals":
        return council_signals(self.h_experts, self.h_mean_dist, step_spans, valid=self.valid)


@dataclass
class CouncilSignals:
    """U_i, V_i và entropy từng expert cho một mẫu."""

    U: torch.Tensor  # (N,)
    V: torch.Tensor  # (N,)
    E: torch.Tensor  # (M, N) - E_i^{(m)}

    def to_numpy(self):
        return {
            "U": self.U.detach().float().cpu().numpy(),
            "V": self.V.detach().float().cpu().numpy(),
            "E": self.E.detach().float().cpu().numpy(),
        }


def council_signals(
    h_experts: torch.Tensor,
    h_mean_dist: torch.Tensor,
    step_spans: Sequence[Span],
    valid: Optional[torch.Tensor] = None,
) -> CouncilSignals:
    """Reduce entropy theo token thành U_i, V_i theo step.

    Args:
        h_experts: ``(M, T)`` — H(phat[m, t]).
        h_mean_dist: ``(T,)`` — H(pbar[t]).
        step_spans: span token trong toạ độ ``input_ids``.
        valid: ``(T,)`` bool — vị trí có entropy hợp lệ.
    """
    n_experts = h_experts.shape[0]
    n_steps = len(step_spans)
    U = h_experts.new_zeros(n_steps)
    V = h_experts.new_zeros(n_steps)
    E = h_experts.new_zeros(n_experts, n_steps)

    for i, (start, end) in enumerate(step_spans):
        start = max(start, 1)  # token 0 không được dự đoán
        end = min(end, h_experts.shape[1])
        if end <= start:
            continue
        if valid is not None:
            mask = valid[start:end]
            if not bool(mask.any()):
                continue
        else:
            mask = None

        h_e = h_experts[:, start:end]
        h_p = h_mean_dist[start:end]
        if mask is not None:
            h_e = h_e[:, mask]
            h_p = h_p[mask]

        e_i = h_e.mean(dim=1)  # (M,)
        E[:, i] = e_i
        U[i] = e_i.mean()
        V[i] = (h_p - h_e.mean(dim=0)).mean()

    V = V.clamp_min(0.0)  # MI >= 0; chặn sai số float âm nhẹ
    return CouncilSignals(U=U, V=V, E=E)


# --------------------------------------------------------------------------- #
# Driver: chạy M expert trên base đóng băng, stream theo chunk token
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_council_signals(
    model,
    adapter_names: Sequence[str],
    input_ids: torch.Tensor,
    step_spans: Sequence[Span],
    *,
    chunk_size: int = 128,
    accumulator_device: Optional[torch.device] = None,
) -> CouncilSignals:
    """Tính U_i, V_i cho một mẫu bằng M LoRA expert trên cùng base đóng băng.

    ``model`` là ``peft.PeftModel`` đã ``load_adapter`` đủ M expert. Ta duyệt
    sequence theo chunk và giữ **một KV cache riêng cho mỗi expert**, nhờ vậy
    tại mỗi thời điểm chỉ có ``(C, V)`` logits của một expert nằm trong bộ nhớ.

    Peak memory ≈ 2 * C * V * 4 bytes (probs hiện tại + accumulator sum_m p),
    không phụ thuộc M hay T.
    """
    model.eval()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    n_experts = len(adapter_names)

    acc = CouncilAccumulator(
        n_experts, seq_len, device=accumulator_device or torch.device("cpu")
    )
    caches: List[object] = [None] * n_experts

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]

        prob_sum: Optional[torch.Tensor] = None
        h_chunk = torch.zeros(n_experts, end - start, dtype=torch.float32, device=device)

        for m, name in enumerate(adapter_names):
            model.set_adapter(name)
            out = model(
                input_ids=chunk,
                past_key_values=caches[m],
                use_cache=True,
            )
            caches[m] = out.past_key_values
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)  # (C, V)
            probs = logp.exp()
            h_chunk[m] = -(probs * logp).sum(dim=-1)
            prob_sum = probs if prob_sum is None else prob_sum.add_(probs)
            del out, logp, probs

        pbar = prob_sum / n_experts
        h_pbar = entropy_from_probs(pbar)
        del prob_sum, pbar

        # logits tại vị trí p dự đoán token p+1 -> shift index sang +1.
        acc.add_chunk(start + 1, h_chunk, h_pbar)
        del h_chunk, h_pbar

    return acc.signals(step_spans)


@torch.no_grad()
def compute_base_step_entropy(
    model,
    input_ids: torch.Tensor,
    step_spans: Sequence[Span],
    *,
    chunk_size: int = 128,
    accumulator_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Length-normalized step entropy dưới **một** model (baseline §6).

    Trả về ``H`` shape ``(N,)``. Dùng cho Entropy Selection baseline, nơi tín
    hiệu chỉ đến từ base student chứ không có hội đồng.
    """
    signals = _single_model_entropy(
        model,
        input_ids,
        step_spans,
        chunk_size=chunk_size,
        accumulator_device=accumulator_device,
    )
    return signals


@torch.no_grad()
def _single_model_entropy(
    model,
    input_ids: torch.Tensor,
    step_spans: Sequence[Span],
    *,
    chunk_size: int,
    accumulator_device: Optional[torch.device],
) -> torch.Tensor:
    model.eval()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    acc = CouncilAccumulator(1, seq_len, device=accumulator_device or torch.device("cpu"))
    cache = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        out = model(input_ids=input_ids[:, start:end], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        h = entropy_from_logits(out.logits[0])  # (C,)
        acc.add_chunk(start + 1, h.unsqueeze(0), h)
        del out, h
    return acc.signals(step_spans).E[0]
