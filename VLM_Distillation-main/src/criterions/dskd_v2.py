from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.criterions.various_divergence import VariousDivergence


def get_hidden_states(outputs) -> Tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("DSKDv2 requires model outputs with hidden_states.")
    return tuple(hidden_states)


def get_output_head(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_output_embeddings"):
        head = encoder.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(encoder, "lm_head"):
        return encoder.lm_head
    raise RuntimeError("Could not find output embedding/lm_head for DSKDv2.")


def call_with_hidden_states(model: nn.Module, inputs: Dict[str, Any]):
    model_inputs = dict(inputs)
    model_inputs.setdefault("output_hidden_states", True)
    return model(**model_inputs)


def project(projector: nn.Module, value: torch.Tensor) -> torch.Tensor:
    target_dtype = next(projector.parameters()).dtype
    return projector(value.to(dtype=target_dtype))


def require_projector(projectors: Any, name: str) -> nn.Module:
    if not isinstance(projectors, nn.ModuleDict) or name not in projectors:
        raise RuntimeError(
            f"DSKDv2 requires a named projector `{name}` in projector_config_path. "
            "Expected projectors: t2s, s2t."
        )
    return projectors[name]


class DSKDv2Criterion(VariousDivergence):
    """Dual-Space KD v2 with ETA, adapted to the VLM distillation API.

    The reference implementation is text-only. This port restricts every KD
    term to shifted assistant text positions using labels and text_feature_mask.
    """

    def __init__(self, args):
        super().__init__(args)
        self.only_stu_kd = bool(getattr(args, "only_stu_kd", False))
        self.only_tea_kd = bool(getattr(args, "only_tea_kd", False))
        self.init_s2t_projector = bool(getattr(args, "init_s2t_projector", False))
        self.t2s_agreement_threshold = float(getattr(args, "t2s_agreement", 1.0))

    def compute_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        shift: bool = True,
    ) -> torch.Tensor:
        if shift:
            logits, target = self.shift_logits_and_labels(logits, target)
        logits = logits.masked_fill(logits.isnan() | logits.isinf(), 0.0)
        target = target.to(device=logits.device)
        if target.numel() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.padding_id,
            reduction="sum",
            label_smoothing=self.label_smoothing,
        )

    def compute_forward_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "sum",
        log: dict | None = None,
        teacher_temperature: float = 1.0,
        shift: bool = False,
    ) -> torch.Tensor:
        student_logits, teacher_logits, target = self._prepare_logits_and_target(
            student_logits,
            teacher_logits,
            target,
            shift=shift,
            teacher_temperature=teacher_temperature,
        )

        student_lprobs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_lprobs = F.log_softmax(teacher_logits, dim=-1)
        per_token = (teacher_probs * (teacher_lprobs - student_lprobs)).sum(dim=-1)
        bad_logits = student_logits.isinf().any(dim=-1) | teacher_logits.isinf().any(dim=-1)
        per_token = per_token.masked_fill(bad_logits, 0.0)
        return self._masked_reduce(per_token, target, reduction, log, "forward_kl")

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running DSKDv2.")

        student_outputs = call_with_hidden_states(distiller.student, student_inputs)
        labels = student_inputs["labels"].to(device=student_outputs.logits.device)
        supervised_loss_sum = self.compute_cross_entropy_loss(student_outputs.logits, labels)
        _shifted_logits, shifted_labels = self.shift_logits_and_labels(student_outputs.logits, labels)
        supervised_loss = supervised_loss_sum / shifted_labels.ne(self.padding_id).sum().float().clamp_min(1.0)

        with torch.no_grad():
            teacher_outputs = call_with_hidden_states(distiller.teacher, teacher_inputs)

        teacher_labels, _teacher_mask = self.teacher_targets(teacher_inputs, student_outputs.logits.device)
        kd_loss, extra = self._compute_dual_space_kd_loss(
            distiller,
            student_inputs,
            teacher_inputs,
            student_outputs,
            teacher_outputs,
            labels,
            teacher_labels,
        )

        loss = (1.0 - self.kd_rate) * supervised_loss + self.kd_rate * kd_loss
        result = {
            "loss": loss,
            "supervised_loss": supervised_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "token_accuracy": self.compute_token_accuracy(student_outputs.logits, labels).detach(),
            "token_correct": self.compute_token_correct(student_outputs.logits, labels).detach(),
            "token_count": self.compute_token_count(student_outputs.logits, labels).detach(),
        }
        result.update({name: value.detach() for name, value in extra.items()})
        return result

    def _compute_dual_space_kd_loss(
        self,
        distiller: Any,
        student_inputs: Dict[str, torch.Tensor],
        teacher_inputs: Dict[str, torch.Tensor],
        student_outputs,
        teacher_outputs,
        labels: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _student_input, target, pad_mask = self.shift_inputs_for_causal_targets(
            student_inputs["input_ids"].to(device=labels.device),
            labels,
        )
        _teacher_input, teacher_target, teacher_pad_mask = self.shift_inputs_for_causal_targets(
            teacher_inputs["input_ids"].to(device=labels.device),
            teacher_labels.to(device=labels.device),
        )

        common_len = min(
            target.shape[1],
            teacher_target.shape[1],
            get_hidden_states(student_outputs)[-1].shape[1],
            get_hidden_states(teacher_outputs)[-1].shape[1],
            student_outputs.logits.shape[1],
            teacher_outputs.logits.shape[1],
        )
        target = target[:, :common_len]
        teacher_target = teacher_target[:, :common_len]
        pad_mask = pad_mask[:, :common_len] & self._shifted_text_mask(student_outputs, common_len, labels.device)
        teacher_pad_mask = teacher_pad_mask[:, :common_len] & self._shifted_text_mask(teacher_outputs, common_len, labels.device)
        target = target.masked_fill(~pad_mask, self.padding_id)
        teacher_target = teacher_target.masked_fill(~teacher_pad_mask, self.padding_id)

        hiddens = get_hidden_states(student_outputs)[-1][:, :common_len]
        teacher_hiddens = get_hidden_states(teacher_outputs)[-1][:, :common_len].to(device=hiddens.device)
        student_logits = student_outputs.logits[:, :common_len]
        teacher_logits = teacher_outputs.logits[:, :common_len].to(device=hiddens.device)
        loss_denom = pad_mask.sum().float().clamp_min(1.0)

        # student space: same position-wise logic as dskdv2.py, restricted to text masks.
        t2s_hiddens = project(require_projector(distiller.projectors, "t2s"), teacher_hiddens)
        student_head = get_output_head(distiller.student)
        student_head_weight = student_head.weight.detach().to(device=t2s_hiddens.device, dtype=t2s_hiddens.dtype)
        t2s_logits = t2s_hiddens.matmul(student_head_weight.transpose(-1, -2))

        t_preds = teacher_logits.argmax(dim=-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
        t_preds_for_ce = t_preds.masked_fill(t_preds.ge(t2s_logits.shape[-1]), self.padding_id)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, t_preds_for_ce, shift=False) / loss_denom

        t2s_agreement_mask = t2s_logits.argmax(dim=-1).eq(t_preds) & pad_mask
        t2s_agreement = t2s_agreement_mask.sum().float() / loss_denom
        t2s_kd_loss = self.dist_func(student_logits, t2s_logits.detach(), target, reduction="none")
        if t2s_agreement <= self.t2s_agreement_threshold:
            t2s_kd_loss = (
                t2s_kd_loss * t2s_agreement_mask.to(dtype=t2s_kd_loss.dtype)
            ).sum() / t2s_agreement_mask.sum().float().clamp_min(1e-3)
        else:
            t2s_kd_loss = (t2s_kd_loss * pad_mask.to(dtype=t2s_kd_loss.dtype)).sum() / loss_denom

        # teacher space: same position-wise logic as dskdv2.py, restricted to text masks.
        s2t_hiddens = self._student_to_teacher_value(distiller, hiddens)
        teacher_head = get_output_head(distiller.teacher)
        s2t_logits = teacher_head(s2t_hiddens.to(dtype=teacher_hiddens.dtype))
        s2t_kd_loss = self.dist_func(s2t_logits, teacher_logits, teacher_target, reduction="none")
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask.to(dtype=s2t_kd_loss.dtype)).sum() / loss_denom

        if self.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        s2t_agreement = (s2t_logits.argmax(dim=-1).eq(student_logits.argmax(dim=-1)) & pad_mask).sum().float() / loss_denom
        t_acc = (teacher_logits.argmax(dim=-1).eq(target) & pad_mask).sum().float() / loss_denom
        t2s_acc = (t2s_logits.argmax(dim=-1).eq(target) & pad_mask).sum().float() / loss_denom

        return kd_loss, {
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "s2t_kd_loss": s2t_kd_loss,
            "t2s_agreement": t2s_agreement,
            "s2t_agreement": s2t_agreement,
            "t_acc": t_acc,
            "t2s_acc": t2s_acc,
            "text_token_count": loss_denom.detach(),
        }

    def _student_to_teacher_value(self, distiller: Any, student_hidden: torch.Tensor) -> torch.Tensor:
        if self.init_s2t_projector and hasattr(distiller, "part_teacher_head_pinv"):
            student_head = get_output_head(distiller.student).weight.detach().transpose(0, 1)
            overlap_ids = getattr(distiller, "student_overlap_token_ids", None)
            if overlap_ids is not None:
                student_head = student_head[:, overlap_ids.to(device=student_head.device)]
            topk_vocab = int(getattr(self.args, "dskd_topk_vocab", getattr(self.args, "topk_vocab", -1)) or -1)
            if topk_vocab != -1:
                student_head = student_head[:, :topk_vocab]
            part_teacher_head_pinv = distiller.part_teacher_head_pinv.to(
                device=student_head.device,
                dtype=student_head.dtype,
            )
            s2t_projector = student_head @ part_teacher_head_pinv
            return student_hidden @ s2t_projector.to(device=student_hidden.device, dtype=student_hidden.dtype)
        return project(require_projector(distiller.projectors, "s2t"), student_hidden)

    def _shifted_text_mask(self, outputs, target_len: int, device: torch.device) -> torch.Tensor:
        text_mask = getattr(outputs, "text_feature_mask", None)
        if text_mask is None:
            return torch.ones(
                get_hidden_states(outputs)[-1].shape[0],
                target_len,
                dtype=torch.bool,
                device=device,
            )
        text_mask = text_mask.to(device=device, dtype=torch.bool)
        shifted = text_mask[:, :target_len]
        if shifted.shape[1] < target_len:
            pad = torch.zeros(shifted.shape[0], target_len - shifted.shape[1], dtype=torch.bool, device=device)
            shifted = torch.cat([shifted, pad], dim=1)
        return shifted
