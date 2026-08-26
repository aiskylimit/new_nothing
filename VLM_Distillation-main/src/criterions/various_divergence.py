from typing import Any, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.criterions.cross_entropy_loss import CrossEntropyLoss


class VariousDivergence(CrossEntropyLoss):
    def __init__(self, args, padding_id: int = -100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.ce_rate = float(getattr(args, "ce_rate", 1.0))
        self.kd_rate = float(getattr(args, "kd_rate", getattr(args, "kd_weight", 1.0)))
        self.dtw_rate = float(getattr(args, "dtw_rate", 0.0))
        self.kd_temperature = float(getattr(args, "kd_temperature", getattr(args, "temperature", 1.0)))
        self.teacher_temperature = float(getattr(args, "teacher_temperature", 1.0))
        self.kd_objective = getattr(args, "kd_objective", "forward_kl").lower()
        
        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")

    def _prepare_logits_and_target(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        shift: bool = False,
        teacher_temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if shift:
            student_logits, target = self.shift_logits_and_labels(student_logits, target)
            teacher_logits = teacher_logits[:, : student_logits.shape[1]]
        dim = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_logits = student_logits[..., :dim] / max(self.kd_temperature, 1e-6)
        teacher_logits = teacher_logits[..., :dim] / max(self.kd_temperature, 1e-6)
        teacher_logits = teacher_logits / max(float(teacher_temperature), 1e-6)
        return student_logits, teacher_logits, target

    def _masked_reduce(
        self,
        per_token: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
        log: dict | None,
        log_name: str,
    ) -> torch.Tensor:
        if reduction == "none":
            return per_token
        if reduction != "sum":
            raise ValueError(f"Unsupported reduction: {reduction}")
        mask = target.to(device=per_token.device).ne(self.padding_id)
        loss = (per_token * mask.to(per_token.dtype)).sum()
        if log is not None:
            log[log_name] = loss
        return loss

    def forward_kl_divergence(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_forward_kl_divergence(*args, **kwargs)

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

        student_lprobs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        teacher_lprobs = F.log_softmax(teacher_logits.float(), dim=-1)
        per_token = (teacher_probs * (teacher_lprobs - student_lprobs)).sum(dim=-1)
        per_token = per_token.masked_fill(student_logits.isinf().any(dim=-1) | teacher_logits.isinf().any(dim=-1), 0.0)
        return self._masked_reduce(per_token, target, reduction, log, "forward_kl")

    def compute_reverse_kl_divergence(
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
        student_probs = F.softmax(student_logits.float(), dim=-1)
        student_lprobs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_lprobs = F.log_softmax(teacher_logits.float(), dim=-1)
        per_token = (student_probs * (student_lprobs - teacher_lprobs)).sum(dim=-1)
        per_token = per_token.masked_fill(student_logits.isinf().any(dim=-1) | teacher_logits.isinf().any(dim=-1), 0.0)
        return self._masked_reduce(per_token, target, reduction, log, "reverse_kl")

    def compute_adaptive_kl_divergence(
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
        alpha = float(getattr(self.args, "adaptive_kl_alpha", 0.5))
        student_probs = F.softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        sorted_teacher_probs, sorted_idx = teacher_probs.sort(dim=-1)
        sorted_student_probs = student_probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_student_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, dim=-1)
        tail_mask = cum_teacher_probs.le(alpha).float()
        g_head = (gap * (1.0 - tail_mask)).sum(dim=-1).detach()
        g_tail = (gap * tail_mask).sum(dim=-1).detach()
        denom = (g_head + g_tail).clamp_min(1e-9)

        fkl = self.compute_forward_kl_divergence(student_logits, teacher_logits, target, reduction="none")
        rkl = self.compute_reverse_kl_divergence(student_logits, teacher_logits, target, reduction="none")
        per_token = (g_head / denom) * fkl + (g_tail / denom) * rkl
        return self._masked_reduce(per_token, target, reduction, log, "adaptive_kl")

    def compute_skewed_forward_kl_divergence(
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
        skew_lambda = float(getattr(self.args, "skew_lambda", 0.5))
        student_probs = F.softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        mixed_probs = skew_lambda * teacher_probs + (1.0 - skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs.clamp_min(1e-9))
        teacher_lprobs = F.log_softmax(teacher_logits.float(), dim=-1)
        per_token = (teacher_probs * (teacher_lprobs - mixed_lprobs)).sum(dim=-1)
        per_token = per_token.masked_fill(student_logits.isinf().any(dim=-1) | teacher_logits.isinf().any(dim=-1), 0.0)
        return self._masked_reduce(per_token, target, reduction, log, "skewed_forward_kl")

    def compute_skewed_reverse_kl_divergence(
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
        skew_lambda = float(getattr(self.args, "skew_lambda", 0.5))
        student_probs = F.softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        mixed_probs = (1.0 - skew_lambda) * teacher_probs + skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs.clamp_min(1e-9))
        student_lprobs = F.log_softmax(student_logits.float(), dim=-1)
        per_token = (student_probs * (student_lprobs - mixed_lprobs)).sum(dim=-1)
        per_token = per_token.masked_fill(student_logits.isinf().any(dim=-1) | teacher_logits.isinf().any(dim=-1), 0.0)
        return self._masked_reduce(per_token, target, reduction, log, "skewed_reverse_kl")

    def compute_js_divergence(
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
        student_probs = F.softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        mixed_probs = 0.5 * (student_probs + teacher_probs)
        student_lprobs = torch.log(student_probs.clamp_min(1e-9))
        teacher_lprobs = torch.log(teacher_probs.clamp_min(1e-9))
        mixed_lprobs = torch.log(mixed_probs.clamp_min(1e-9))
        per_token = 0.5 * (
            (teacher_probs * (teacher_lprobs - mixed_lprobs)).sum(dim=-1)
            + (student_probs * (student_lprobs - mixed_lprobs)).sum(dim=-1)
        )
        per_token = per_token.masked_fill(student_logits.isinf().any(dim=-1) | teacher_logits.isinf().any(dim=-1), 0.0)
        return self._masked_reduce(per_token, target, reduction, log, "js_divergence")

    @staticmethod
    def student_entropy_weights(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        weights = entropy.detach() * mask.float()
        tokens = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return weights * (tokens / weights.sum(dim=1, keepdim=True).clamp_min(1e-9))

    @staticmethod
    def teacher_certainty_weights(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        certainty = (1.0 - entropy / (math.log(logits.shape[-1]) + 1e-9)).clamp(min=0.0, max=1.0)
        weights = certainty.detach() * mask.float()
        tokens = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return weights * (tokens / weights.sum(dim=1, keepdim=True).clamp_min(1e-9))
