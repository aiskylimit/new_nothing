from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import print_master


def _masked_mean(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)

    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype)
    if mask.shape[-1] != hidden.shape[1]:
        min_len = min(mask.shape[-1], hidden.shape[1])
        mask = mask[:, -min_len:]
        hidden = hidden[:, -min_len:, :]

    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom


def _project(projector, value: torch.Tensor) -> torch.Tensor:
    target_dtype = next(projector.parameters()).dtype
    projected = projector(value.to(dtype=target_dtype))
    return projected.to(dtype=torch.float32)


def _layer_hidden_states(outputs, mapping):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "The model did not return hidden_states. Distillation with "
            "teacher_layer_mapping/student_layer_mapping requires hidden states."
        )
    return hidden_states[mapping]


class DefaultDistillationCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kd_weight = float(getattr(args, "kd_weight", 0.0) or 0.0)

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> torch.Tensor:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running teacher distillation.")

        student_outputs = distiller.student(**student_inputs)
        ce_loss = student_outputs.loss
        if ce_loss is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        kd_terms = []
        student_layers = list(getattr(distiller.training_args, "student_layer_mapping", []) or [])
        teacher_layers = list(getattr(distiller.training_args, "teacher_layer_mapping", []) or [])

        if student_layers and teacher_layers:
            if len(student_layers) != len(teacher_layers):
                raise ValueError("student_layer_mapping and teacher_layer_mapping must have the same length.")
            if len(student_layers) > len(distiller.projectors):
                raise ValueError("Layer mappings require at least one projector per mapped layer.")

            for idx, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
                student_hidden = _layer_hidden_states(student_outputs, student_layer)
                teacher_hidden = _layer_hidden_states(teacher_outputs, teacher_layer)

                student_vec = _masked_mean(student_hidden, student_inputs.get("attention_mask"))
                teacher_vec = _masked_mean(teacher_hidden, teacher_inputs.get("attention_mask"))
                student_vec = _project(distiller.projectors[idx], student_vec)
                teacher_vec = teacher_vec.to(device=student_vec.device, dtype=torch.float32)
                kd_terms.append(F.mse_loss(student_vec, teacher_vec))

        else:
            student_logits = getattr(student_outputs, "logits", None)
            teacher_logits = getattr(teacher_outputs, "logits", None)
            if student_logits is not None and teacher_logits is not None:
                min_seq_len = min(student_logits.shape[1], teacher_logits.shape[1])
                min_logit_dim = min(student_logits.shape[-1], teacher_logits.shape[-1])
                student_logits = student_logits[:, -min_seq_len:, :min_logit_dim].float()
                teacher_logits = teacher_logits[:, -min_seq_len:, :min_logit_dim].to(student_logits.device).float()

                labels = student_inputs.get("labels")
                if labels is not None:
                    labels = labels[:, -min_seq_len:].to(student_logits.device)
                    token_mask = labels.ne(-100)
                else:
                    token_mask = student_inputs.get(
                        "attention_mask",
                        torch.ones(student_logits.shape[:2], device=student_logits.device),
                    )
                    token_mask = token_mask[:, -min_seq_len:].bool()

                temperature = max(float(getattr(distiller.model_args, "temperature", 1.0) or 1.0), 1e-6)
                kl = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                kd_terms.append((kl * token_mask.to(kl.dtype)).sum() / token_mask.sum().clamp_min(1))
            elif not getattr(distiller, "_warned_no_usable_kd", False):
                print_master(
                    "Teacher is enabled, but no layer mapping/projector was provided and logits "
                    "shapes are incompatible. Training will use student CE loss only."
                )
                distiller._warned_no_usable_kd = True

        kd_loss = sum(kd_terms) / len(kd_terms) if kd_terms else ce_loss.new_zeros(())
        return ce_loss + self.kd_weight * kd_loss


def default_distillation_criterion(distiller: Any, batch: Dict[str, Any]) -> torch.Tensor:
    return DefaultDistillationCriterion(distiller.training_args)(distiller, batch)
