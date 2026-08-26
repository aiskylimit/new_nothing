from typing import Any, Dict

import torch
import torch.nn as nn


class CEOnlyCriterion(nn.Module):
    """Teacher-mode supervised baseline."""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_outputs = distiller.student(**batch["student_inputs"])
        loss = student_outputs.loss
        if loss is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")
        return {
            "loss": loss,
            "hard_loss": loss.detach(),
        }
