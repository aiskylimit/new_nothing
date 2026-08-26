from typing import Any, Dict

import torch
import torch.nn as nn

from src.criterions.em_kd import EMKDCriterion
from src.criterions.sre import SRECriterion


class UnitAlignedDistillationCriterion(nn.Module):
    """Joint SRE + EMKD criterion for unit-aligned VLM distillation."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sre = SRECriterion(args)
        self.emkd = EMKDCriterion(args)
        self.ce_weight = float(getattr(args, "joint_ce_weight", 0.5))
        self.sre_weight = float(getattr(args, "joint_sre_weight", 1.0))
        self.emkd_weight = float(getattr(args, "joint_emkd_weight", 1.0))

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running joint unit-aligned distillation.")

        student_outputs = distiller.student(**student_inputs)
        hard_loss = student_outputs.loss
        if hard_loss is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        sre_kd_loss, sre_span_loss, sre_geom_loss, sre_logit_loss = self.sre._knowledge_distillation_loss(
            distiller,
            student_outputs,
            teacher_outputs,
            student_inputs,
            teacher_inputs,
        )
        response_loss = self.emkd._response_logit_distillation(
            student_outputs,
            teacher_outputs,
            student_inputs,
        )
        vision_loss, vision_logit_loss, affinity_loss, matched_count = self.emkd._vision_distillation(
            distiller,
            student_outputs,
            teacher_outputs,
        )

        emkd_kd_loss = (1.0 - self.emkd.alpha) * response_loss + vision_loss
        kd_weight_sum = max(self.sre_weight + self.emkd_weight, 1e-6)
        joint_kd_loss = (
            self.sre_weight * sre_kd_loss
            + self.emkd_weight * emkd_kd_loss
        ) / kd_weight_sum

        ce_weight = min(max(self.ce_weight, 0.0), 1.0)
        loss = ce_weight * hard_loss + (1.0 - ce_weight) * joint_kd_loss

        return {
            "loss": loss,
            "hard_loss": hard_loss.detach(),
            "joint_kd_loss": joint_kd_loss.detach(),
            "sre_kd_loss": sre_kd_loss.detach(),
            "sre_span_loss": sre_span_loss.detach(),
            "sre_geom_loss": sre_geom_loss.detach(),
            "sre_logit_loss": sre_logit_loss.detach(),
            "emkd_kd_loss": emkd_kd_loss.detach(),
            "response_kd_loss": response_loss.detach(),
            "vision_kd_loss": vision_loss.detach(),
            "vision_logit_loss": vision_logit_loss.detach(),
            "affinity_loss": affinity_loss.detach(),
            "matched_vision_samples": hard_loss.new_tensor(float(matched_count)),
        }
