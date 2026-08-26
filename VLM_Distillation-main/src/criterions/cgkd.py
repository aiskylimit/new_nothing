"""Confidence-Gated Generative-distribution KD (CGKD).

Standard logit distillation aligns the full next-token distributions of teacher
and student:
    KL( P_T^(t) || P_S^(t) )
Not every teacher prediction is equally reliable, though. When the teacher's
own distribution is high-entropy (uncertain), forcing the student to match it
transfers noise. CGKD preserves the full vocabulary (no top-k cropping) and
instead *gates* the KD strength by the teacher's normalized entropy:

    h̃_t = H(P_T^(t)) / log|V|
    w_t  = exp(-h̃_t)                # high w when teacher is confident
    L_CGKD = ( Σ_t w_t · KL(P_T^(t) || P_S^(t)) ) / Σ_t w_t

Per draft.pdf §CGKD.

Single-method criterion mixes with supervised CE in the EM-KD / SRE style:
    L = α · L_CE + (1 - α) · w · L_CGKD
The joint draft criterion (SCVA + CGKD) calls ``_cgkd_loss`` for the pure KD
term and follows the draft's additive formula.

Vocab handling: when student and teacher have different vocab sizes (cross-
tokenizer pairs), we crop to the min of the two — matching how EM-KD's
response distillation handles vocab mismatch. Sequence length is also cropped
defensively.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


def _crop_seq_and_last_dim(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq = min(a.shape[1], b.shape[1])
    vocab = min(a.shape[-1], b.shape[-1])
    return a[:, -seq:, :vocab], b[:, -seq:, :vocab], seq


class CGKDCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = float(getattr(args, "cgkd_alpha", 0.5))
        self.weight = float(getattr(args, "cgkd_weight", 1.0))
        self.temperature = max(float(getattr(args, "cgkd_temperature", 1.0)), 1e-6)

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running CGKD.")

        student_outputs = distiller.student(**student_inputs)
        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        return self.compute_losses(distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs)

    def compute_losses(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        ce = student_outputs.loss
        if ce is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        cgkd_kd = self._cgkd_loss(student_outputs, teacher_outputs, student_inputs)
        total = self.alpha * ce + (1.0 - self.alpha) * self.weight * cgkd_kd
        return {
            "loss": total,
            "hard_loss": ce.detach(),
            "cgkd_loss": cgkd_kd.detach(),
        }

    def _cgkd_loss(
        self,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        s_logits = getattr(student_outputs, "logits", None)
        t_logits = getattr(teacher_outputs, "logits", None)
        if s_logits is None or t_logits is None:
            raise RuntimeError("CGKD requires both student and teacher logits.")

        s_logits, t_logits, min_seq = _crop_seq_and_last_dim(s_logits, t_logits)

        # Response gating: prefer labels mask; fall back to attention mask if absent.
        labels = student_inputs.get("labels")
        if labels is not None:
            response_mask = labels[:, -min_seq:].to(s_logits.device).ne(IGNORE_INDEX)
        elif student_inputs.get("attention_mask") is not None:
            response_mask = student_inputs["attention_mask"][:, -min_seq:].to(s_logits.device).bool()
        else:
            response_mask = torch.ones(s_logits.shape[:2], dtype=torch.bool, device=s_logits.device)

        if response_mask.sum() == 0:
            return s_logits.new_zeros(())

        T = self.temperature
        V = s_logits.shape[-1]

        t_log_probs = F.log_softmax(t_logits.float() / T, dim=-1)
        t_probs = t_log_probs.exp()
        # Per-position teacher entropy / log|V| → normalized in [0, 1].
        h_t = -(t_probs * t_log_probs).sum(dim=-1)
        h_norm = h_t / max(math.log(max(V, 2)), 1e-6)
        w = (-h_norm).exp()                                          # [B, L]

        s_log_probs = F.log_softmax(s_logits.float() / T, dim=-1)
        # KL(P_T || P_S) per position, summed over vocab.
        kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)

        w_masked = w * response_mask.to(dtype=w.dtype)
        weighted = (w_masked * kl).sum() / w_masked.sum().clamp_min(1e-6)
        return weighted * (T ** 2)
