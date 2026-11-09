"""Self-Exploratory Distillation (SED) — CurioSFT eq. (5), (6).

Bài: *Learning While Staying Curious* (ACL 2026), arXiv:2602.02244.
Repo: github.com/HaoooWang/CurioSFT — `custom_verl/recipe/curio_sft/`.

Ý tưởng: ngoài NLL trên token teacher, kéo student về gần một teacher **đã
scale nhiệt độ per-token** (entropy cao hơn, GIỮ NGUYÊN thứ tự token) bằng K2-loss.
Chỉ đẩy exploration ở token entropy cao (reasoning connector), giữ nguyên token
entropy thấp (factual) — nhiệt độ do EGTS (`vendor/curiosft/tau_search.py`) chọn.

    L_SED_t = 1/2 ( log p_student(y_t; τ=1)  -  log p_teacher(y_t; τ̂_t) )²      # eq. (5)
    L_total = L_SFT + α · L_SED                                                  # eq. (6), α=1

Đối chiếu paper ↔ repo (những chỗ dễ nhầm):
  - Repo `core_algos.py::custom_loss` nhánh `curio_sft`:
        mse = -log_prob ;  kd = α·0.5·(log_prob - ref_log_prob)²  ;  loss = masked_mean(mse+kd)
    ⇒ dùng **mean** trên token response (paper eq. 5 viết Σ_t — chỉ khác hệ số LR).
  - `log_prob` / `ref_log_prob` là log-prob của **token đã sinh** y_t (không phải full KL).
  - `ref_log_prob` lấy từ teacher mà logits ĐÃ bị chia cho τ̂_t per-token
    (`dp_actor.py:201-214`), và τ̂ = τ_min ở token có entropy full-vocab ≤ 0.03 (dòng 205).
  - Teacher của repo = bản EMA (verl `sync_ref_model`, µ=0.99, mỗi 5 step). **Ở đây
    ta bỏ EMA** — teacher = base đóng băng (adapter tắt). Xem `--sed_teacher`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from core.council import entropy_from_logits
from vendor.curiosft.tau_search import compute_tau_from_logits

Span = Tuple[int, int]

_TAU_MIN_DEFAULT = 1.1


def k2_step_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    step_spans: Sequence[Span],
) -> torch.Tensor:
    """K2-loss theo step: ``L_i = mean_{t∈step i} 0.5 (logp_s[t] - logp_t[t])²``.

    Cả hai input là vector ``(T,)`` log-prob của **token thực tế** ``input_ids[t]``
    (toạ độ ``input_ids``, phần tử 0 không hợp lệ) — đúng quy ước
    :func:`core.losses.token_logprobs`. Gộp theo step giống
    :func:`core.losses.step_nll_from_logprobs` để cộng thẳng vào vector NLL.
    """
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            f"student {tuple(student_logprobs.shape)} != teacher {tuple(teacher_logprobs.shape)}"
        )
    per_tok = 0.5 * (student_logprobs - teacher_logprobs.to(student_logprobs.dtype)) ** 2
    out = per_tok.new_zeros(len(step_spans))
    seq_len = per_tok.shape[0]
    for i, (start, end) in enumerate(step_spans):
        start = max(start, 1)  # vị trí 0 không có logprob
        end = min(end, seq_len)
        if end <= start:
            continue
        out[i] = per_tok[start:end].mean()
    return out


@torch.no_grad()
def teacher_sed_logprobs(
    model,
    input_ids: torch.Tensor,
    *,
    adaptive_tau: bool = True,
    fixed_tau: float = 1.3,
    disable_adapter: bool = True,
    chunk_size: int = 256,
    tau_min: float = 1.1,
    tau_max: float = 1.5,
    egts_topk: int = 512,
    egts_iters: int = 20,
    h_pivot: float = 1.2,
    gamma: float = 2.0,
    delta_max: float = 0.5,
    low_entropy_cutoff: float = 0.03,
    return_tau: bool = False,
) -> torch.Tensor:
    """log p_teacher(input_ids[t] | input_ids[:t]) dưới teacher đã scale nhiệt độ.

    Teacher forward chạy **streaming theo chunk token** (KV cache, giống
    :func:`core.council.compute_council_signals`) nên peak bộ nhớ chỉ ~ một
    ``(chunk_size, V)`` logits, không phụ thuộc độ dài chuỗi.

    Args:
        model: ``peft.PeftModel`` đang train. Nếu ``disable_adapter`` → teacher là
            base thuần (mọi LoRA tắt). Nếu ``False`` → teacher = student hiện tại,
            detached (ứng với ablation "w/o Separate Teacher" của bài).
        adaptive_tau: ``True`` → EGTS per-token (eq. 7). ``False`` → ``fixed_tau``
            cho mọi token (ablation "w/o Adaptive Temp.", bài dùng 1.3).
        chunk_size: token/chunk khi stream teacher forward.

    Returns:
        ``(T,)`` float32, phần tử 0 = 0. Nếu ``return_tau`` trả thêm ``tau (T,)``.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    teacher_lp = torch.zeros(seq_len, dtype=torch.float32, device=device)
    tau_full = torch.ones(seq_len, dtype=torch.float32, device=device)

    was_training = model.training
    model.eval()
    ctx = model.disable_adapter() if disable_adapter else _nullcontext()
    try:
        with ctx:
            cache = None
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = input_ids[:, start:end]
                out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0]  # (C, V)

                # logits[p] (vị trí tuyệt đối start+p) dự đoán token start+p+1.
                first_target = start + 1
                targets = input_ids[0, first_target : first_target + logits.shape[0]]
                c = targets.shape[0]
                if c == 0:
                    del out, logits
                    continue
                rows = logits[:c].float()  # (C', V)

                if adaptive_tau:
                    _, tau = compute_tau_from_logits(
                        rows,
                        tau_min=tau_min,
                        tau_max=tau_max,
                        topk=egts_topk,
                        iters=egts_iters,
                        h_pivot=h_pivot,
                        gamma=gamma,
                        delta_max=delta_max,
                        apply_low_entropy_cutoff=False,
                    )
                    # cutoff bằng entropy FULL-VOCAB (khớp dp_actor.py:205)
                    h_full = entropy_from_logits(rows)
                    tau = torch.where(h_full <= low_entropy_cutoff, tau.new_full((), tau_min), tau)
                    scaled = rows / tau.unsqueeze(-1)
                else:
                    tau = rows.new_full((c,), float(fixed_tau))
                    scaled = rows / fixed_tau

                lp = torch.log_softmax(scaled, dim=-1)
                lp = lp.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
                teacher_lp[first_target : first_target + c] = lp
                tau_full[first_target : first_target + c] = tau
                del out, logits, rows, scaled, lp
    finally:
        if was_training:
            model.train()

    if return_tau:
        return teacher_lp, tau_full
    return teacher_lp


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
