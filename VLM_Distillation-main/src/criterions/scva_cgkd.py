"""SCVA + CGKD joint distillation — the draft.pdf headline method.

Total loss (draft notation, additive — NOT the (α, 1-α) split used by
EM-KD/SRE single-method criteria):

    L = ce_weight · L_CE + λ_v · L_SCVA + λ_g · L_CGKD

A single shared student/teacher forward pass per step is used: each
sub-criterion exposes a private ``_*_loss(student_outputs, teacher_outputs,
…)`` method that operates on already-computed outputs, so the joint criterion
does not pay 2× compute.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.criterions.cgkd import CGKDCriterion
from src.criterions.scva import SCVACriterion


class SCVACGKDCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scva = SCVACriterion(args)
        self.cgkd = CGKDCriterion(args)
        self.ce_weight = float(getattr(args, "scva_cgkd_ce_weight", 1.0))
        self.lambda_v = float(getattr(args, "scva_cgkd_lambda_v", 1.0))
        self.lambda_g = float(getattr(args, "scva_cgkd_lambda_g", 1.0))

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running SCVA+CGKD.")

        student_outputs = distiller.student(**student_inputs)
        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        ce = student_outputs.loss
        if ce is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        scva_kd = self.scva._scva_loss(student_outputs, teacher_outputs, student_inputs, teacher_inputs)
        cgkd_kd = self.cgkd._cgkd_loss(student_outputs, teacher_outputs, student_inputs)

        total = self.ce_weight * ce + self.lambda_v * scva_kd + self.lambda_g * cgkd_kd
        out: Dict[str, torch.Tensor] = {
            "loss": total,
            "hard_loss": ce.detach(),
            "scva_loss": scva_kd.detach(),
            "cgkd_loss": cgkd_kd.detach(),
        }
        # Propagate SCVA validity counters set by SCVACriterion._scva_loss above.
        out.update(self.scva._scalarize_counts(ce))
        return out
