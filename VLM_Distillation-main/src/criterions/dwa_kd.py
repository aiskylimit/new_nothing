from typing import Any, Dict, Tuple

import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.criterions.soft_dtw_cuda import SoftDTW
from src.criterions.various_divergence import (
    VariousDivergence,
)

import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("llvmlite").setLevel(logging.WARNING)

def get_hidden_states(outputs) -> Tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("DWA-KD requires model outputs with hidden_states.")
    return tuple(hidden_states)


def get_output_head(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_output_embeddings"):
        head = encoder.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(encoder, "lm_head"):
        return encoder.lm_head
    raise RuntimeError("Could not find output embedding/lm_head for DWA-KD.")


def get_input_embeddings(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_input_embeddings"):
        embeddings = encoder.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    if hasattr(encoder, "model") and hasattr(encoder.model, "embed_tokens"):
        return encoder.model.embed_tokens
    if hasattr(encoder, "model") and hasattr(encoder.model, "model") and hasattr(encoder.model.model, "embed_tokens"):
        return encoder.model.model.embed_tokens
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "wte"):
        return encoder.transformer.wte
    raise RuntimeError("Could not find input embeddings for DWA-KD.")


def project(projector: nn.Module, value: torch.Tensor) -> torch.Tensor:
    target_dtype = next(projector.parameters()).dtype
    projected = projector(value.to(dtype=target_dtype))
    return projected.to(dtype=torch.float32)


def require_projector(projectors: Any, name: str) -> nn.Module:
    if not isinstance(projectors, nn.ModuleDict) or name not in projectors:
        raise RuntimeError(
            f"DWA-KD requires a named projector `{name}` in projector_config_path. "
            "Expected projectors: query, s2t, t2s, dtw_embed_t2s."
        )
    return projectors[name]


def safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.float().std().clamp_min(1e-6)


def cosine_cost_matrices(
    student_seq: torch.Tensor,
    teacher_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DWA cosine costs without broadcasting to [S, T, hidden_dim]."""
    student_norm = F.normalize(student_seq, p=2, dim=-1, eps=1e-8)
    teacher_norm = F.normalize(teacher_seq, p=2, dim=-1, eps=1e-8)
    cross_cost = 1.0 - torch.matmul(student_norm, teacher_norm.transpose(0, 1))
    student_cost = 1.0 - torch.matmul(student_norm, student_norm.transpose(0, 1))
    teacher_cost = 1.0 - torch.matmul(teacher_norm, teacher_norm.transpose(0, 1))
    return cross_cost, student_cost, teacher_cost


class DWAKDCriterion(VariousDivergence):
    """
    DWA-KD criterion adapted to the current VLM_Distill criterion API.

    This module only imports reusable criterion utilities from ``src`` and does
    not depend on the temporary ``dwa_kd`` reference folder.
    """

    def __init__(self, args):
        super().__init__(args)
        self.kd_warmup_steps = int(getattr(args, "kd_warmup_steps", 300) or 0)
        self.dtw_warmup_steps = int(getattr(args, "dtw_warmup_steps", 0) or 0)
        self.dtw_gamma_start = float(getattr(args, "dtw_gamma_start", getattr(args, "dtw_gamma", 2.0)))
        self.dtw_gamma_end = float(getattr(args, "dtw_gamma_end", 0.8))
        self.dtw_gamma_steps = int(getattr(args, "dtw_gamma_steps", 3570) or 0)
        self.dtw_band_width = float(getattr(args, "dtw_band_width", 5))
        self.dtw_band_penalty = float(getattr(args, "dtw_band_penalty", 1.0))
        self.dtw_band_center_blend = float(getattr(args, "dtw_band_center_blend", 0.7))
        self.dtw_band_entropy_coef = float(getattr(args, "dtw_band_entropy_coef", 2.0))
        self.dtw_band_warmup_steps = int(getattr(args, "dtw_band_warmup_steps", 0) or 0)
        self.dtw_band_source = getattr(args, "dtw_band_source", "cma")
        self.only_save_projector = bool(getattr(args, "only_save_projector", False))
        self._global_step = 0
        self.dtw = SoftDTW(use_cuda=False, gamma=float(getattr(args, "dtw_gamma", 2.0))) if self.dtw_rate > 0 else None
        self.last_align = None

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running DWA-KD.")

        student_outputs = distiller.student(**student_inputs)
        labels = student_inputs["labels"].to(device=student_outputs.logits.device)
        supervised_loss = self.compute_cross_entropy_loss(student_outputs.logits, labels)

        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        teacher_labels, teacher_mask = self.teacher_targets(teacher_inputs, student_outputs.logits.device)
        kd_loss, extra = self._dual_space_kd_loss(
            distiller,
            student_inputs,
            teacher_inputs,
            student_outputs,
            teacher_outputs,
            labels,
            teacher_labels,
            teacher_mask,
        )

        self._update_dtw_gamma()

        dtw_loss = self._dtw_loss(
            distiller,
            student_outputs,
            teacher_outputs,
            labels,
            teacher_labels,
            teacher_mask,
        )
        weighted_dtw_loss = dtw_loss * self._warmup_scale(self.dtw_warmup_steps)

        loss = self.ce_rate * supervised_loss + self.kd_rate * kd_loss + self.dtw_rate * weighted_dtw_loss
        self._global_step += 1

        result = {
            "loss": loss,
            "supervised_loss": supervised_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "dtw_loss": dtw_loss.detach(),
            "weighted_dtw_loss": weighted_dtw_loss.detach(),
            "token_accuracy": self.compute_token_accuracy(student_outputs.logits, labels).detach(),
            "token_correct": self.compute_token_correct(student_outputs.logits, labels).detach(),
            "token_count": self.compute_token_count(student_outputs.logits, labels).detach(),
        }
        result.update({name: value.detach() for name, value in extra.items()})
        return result

    def _dual_space_kd_loss(
        self,
        distiller: Any,
        student_inputs: Dict[str, torch.Tensor],
        teacher_inputs: Dict[str, torch.Tensor],
        student_outputs,
        teacher_outputs,
        labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_input, target, student_mask = self.shift_inputs_for_causal_targets(
            student_inputs["input_ids"].to(device=labels.device),
            labels,
        )
        teacher_input, teacher_target, teacher_mask = self.shift_inputs_for_causal_targets(
            teacher_inputs["input_ids"].to(device=labels.device),
            teacher_labels.to(device=labels.device),
        )
        student_text_mask = self._shifted_text_mask(student_outputs, target.shape[1], student_mask.device)
        teacher_text_mask = self._shifted_text_mask(teacher_outputs, teacher_target.shape[1], student_mask.device)
        student_mask = student_mask & student_text_mask
        teacher_mask = teacher_mask & teacher_text_mask
        target = target.masked_fill(~student_mask, self.padding_id)
        teacher_target = teacher_target.masked_fill(~teacher_mask, self.padding_id)
        teacher_mask = teacher_mask.to(device=student_mask.device)
        student_hidden = get_hidden_states(student_outputs)[-1][:, : target.shape[1]]
        teacher_hidden = get_hidden_states(teacher_outputs)[-1][:, : teacher_target.shape[1]].to(device=student_hidden.device)
        student_logits = student_outputs.logits[:, : target.shape[1]]
        teacher_logits = teacher_outputs.logits[:, : teacher_target.shape[1]].to(device=student_hidden.device)

        student_weight = self.student_entropy_weights(student_logits, student_mask)
        teacher_weight = self.teacher_certainty_weights(teacher_logits, teacher_mask)

        student_embed = get_input_embeddings(distiller.student)
        teacher_embed = get_input_embeddings(distiller.teacher)

        formal_labels = torch.where(student_mask, target, torch.zeros_like(target))
        formal_student_input = torch.where(student_mask, student_input, torch.zeros_like(student_input))
        formal_teacher_labels = torch.where(teacher_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_mask, teacher_input, torch.zeros_like(teacher_input))

        student_input_embeds = student_embed(formal_student_input).detach()
        student_target_embeds = student_embed(formal_labels).detach()
        teacher_input_embeds = teacher_embed(formal_teacher_input).detach().to(device=student_hidden.device)
        teacher_target_embeds = teacher_embed(formal_teacher_labels).detach().to(device=student_hidden.device)

        student_index_embeds = torch.cat([student_input_embeds, student_target_embeds], dim=-1)
        teacher_index_embeds = torch.cat([teacher_input_embeds, teacher_target_embeds], dim=-1)

        student_query = project(require_projector(distiller.projectors, "query"), student_index_embeds)
        teacher_key = (teacher_index_embeds / safe_std(teacher_index_embeds)).float()
        student_value = project(require_projector(distiller.projectors, "s2t"), student_hidden)
        teacher_value = project(
            require_projector(distiller.projectors, "t2s"),
            teacher_hidden / safe_std(teacher_hidden) + teacher_target_embeds / safe_std(teacher_target_embeds),
        )

        align = torch.matmul(student_query, teacher_key.transpose(-1, -2))
        align = align / math.sqrt(max(float(teacher_hidden.shape[-1] * 2), 1.0))
        align_mask = student_mask.float().unsqueeze(-1) * teacher_mask.float().unsqueeze(1)
        align = align.masked_fill(align_mask.eq(0), -100000.0)

        t2s_weight = torch.softmax(align, dim=-1)
        self.last_align = t2s_weight.detach()
        t2s_hidden = torch.matmul(t2s_weight, teacher_value).to(dtype=student_hidden.dtype)
        student_head = get_output_head(distiller.student)
        t2s_logits = student_head(t2s_hidden)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target, shift=False)

        if self.only_save_projector:
            return t2s_ce_loss, {"t2s_ce_loss": t2s_ce_loss}

        t2s_kd_vec = self.dist_func(
            student_logits,
            t2s_logits.detach(),
            target,
            teacher_temperature=self.teacher_temperature,
            reduction="none",
        )
        t2s_confidence = torch.softmax(t2s_logits.detach().float(), dim=-1).max(dim=-1)[0]
        if self.kd_warmup_steps > 0:
            t2s_confidence = t2s_confidence * min(1.0, float(self._global_step + 1) / float(self.kd_warmup_steps))
        t2s_gate = t2s_confidence * student_mask.float()
        t2s_kd_loss = (t2s_kd_vec * t2s_gate).sum()
        t2s_kd_loss_weighted = (t2s_kd_vec * t2s_gate * student_weight).sum()

        s2t_weight = torch.softmax(align.transpose(-1, -2), dim=-1)
        s2t_hidden = torch.matmul(s2t_weight, student_value).to(dtype=teacher_hidden.dtype)
        teacher_head = get_output_head(distiller.teacher)
        s2t_logits = teacher_head(s2t_hidden)
        s2t_kd_vec = self.dist_func(
            s2t_logits,
            teacher_logits.to(device=s2t_logits.device),
            teacher_target,
            reduction="none",
        )
        s2t_kd_loss = (s2t_kd_vec * teacher_mask.float()).sum()
        s2t_kd_loss_weighted = (s2t_kd_vec * teacher_mask.float() * teacher_weight).sum()
        kd_loss = t2s_ce_loss + t2s_kd_loss_weighted + s2t_kd_loss_weighted

        return kd_loss, {
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "s2t_kd_loss": s2t_kd_loss,
        }

    def _dtw_loss(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.dtw is None:
            return student_outputs.logits.new_zeros(())

        _, target, student_mask = self.shift_inputs_for_causal_targets(
            torch.zeros_like(labels),
            labels,
        )
        _, teacher_target, teacher_mask = self.shift_inputs_for_causal_targets(
            torch.zeros_like(teacher_labels),
            teacher_labels.to(device=labels.device),
        )
        student_text_mask = self._shifted_text_mask(student_outputs, target.shape[1], student_mask.device)
        teacher_text_mask = self._shifted_text_mask(teacher_outputs, teacher_target.shape[1], student_mask.device)
        student_mask = student_mask & student_text_mask
        teacher_mask = teacher_mask & teacher_text_mask
        target = target.masked_fill(~student_mask, self.padding_id)
        teacher_target = teacher_target.masked_fill(~teacher_mask, self.padding_id)
        student_embed = get_input_embeddings(distiller.student)
        teacher_embed = get_input_embeddings(distiller.teacher)
        formal_labels = torch.where(student_mask, target, torch.zeros_like(target))
        formal_teacher_labels = torch.where(teacher_mask, teacher_target, torch.zeros_like(teacher_target))
        student_target_embeds = student_embed(formal_labels)
        teacher_target_embeds = teacher_embed(formal_teacher_labels).detach().to(device=student_target_embeds.device)

        student_hidden = get_hidden_states(student_outputs)[-1][:, : target.shape[1]]
        teacher_hidden = get_hidden_states(teacher_outputs)[-1][:, : teacher_target.shape[1]].to(device=student_hidden.device)
        projected_teacher_embeds = project(require_projector(distiller.projectors, "dtw_embed_t2s"), teacher_target_embeds)
        hidden_projector_name = "dtw_hidden_t2s" if "dtw_hidden_t2s" in distiller.projectors else "t2s"
        projected_teacher_hidden = project(require_projector(distiller.projectors, hidden_projector_name), teacher_hidden)

        embed_loss = self._alignment_loss(student_target_embeds, projected_teacher_embeds, student_mask, teacher_mask)
        hidden_loss = self._alignment_loss(student_hidden, projected_teacher_hidden, student_mask, teacher_mask)
        return embed_loss + hidden_loss

    def _alignment_loss(
        self,
        student_embs: torch.Tensor,
        teacher_embs: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> torch.Tensor:
        total = student_embs.new_zeros(())
        pairs = 0

        for index in range(student_embs.shape[0]):
            student_positions = student_mask[index].nonzero(as_tuple=False).flatten()
            teacher_positions = teacher_mask[index].nonzero(as_tuple=False).flatten()
            student_len = int(student_positions.numel())
            teacher_len = int(teacher_positions.numel())
            if student_len == 0 or teacher_len == 0:
                continue
            pairs += 1
            student_seq = student_embs[index].index_select(0, student_positions).float()
            teacher_seq = teacher_embs[index].index_select(0, teacher_positions).float()
            cross_cost, student_cost, teacher_cost = cosine_cost_matrices(student_seq, teacher_seq)

            cross_cost = self._apply_adaptive_band(cross_cost, index, student_positions, teacher_positions)
            s2t = self.dtw.forward_with_cost_matrix(cross_cost.unsqueeze(0))
            s2s = self.dtw.forward_with_cost_matrix(student_cost.unsqueeze(0))
            t2t = self.dtw.forward_with_cost_matrix(teacher_cost.unsqueeze(0))
            total = total + (s2t - 0.5 * (s2s + t2t)).squeeze()

        if pairs == 0:
            return student_embs.new_zeros(())
        return total

    def _apply_adaptive_band(
        self,
        cross_cost: torch.Tensor,
        batch_index: int,
        student_positions: torch.Tensor,
        teacher_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.dtw_band_width <= 0:
            return cross_cost
        if self.dtw_band_source != "cma" or self.last_align is None:
            return cross_cost

        student_len = int(student_positions.numel())
        teacher_len = int(teacher_positions.numel())
        align = self.last_align[batch_index].index_select(0, student_positions).index_select(1, teacher_positions)
        align = (align + 1e-9) / align.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        row_entropy = -(align * torch.log(align)).sum(dim=-1)
        linear_center = torch.arange(student_len, device=align.device, dtype=torch.float32) * (float(teacher_len) / float(student_len))
        soft_center = (align * torch.arange(teacher_len, device=align.device, dtype=torch.float32).view(1, -1)).sum(dim=-1)
        alpha = self.dtw_band_center_blend
        centers = alpha * soft_center + (1.0 - alpha) * linear_center
        width = self.dtw_band_width + self.dtw_band_entropy_coef * row_entropy
        teacher_positions = torch.arange(teacher_len, device=align.device, dtype=torch.float32).view(1, -1)
        band = (teacher_positions - centers.view(-1, 1)).abs() <= width.view(-1, 1)
        return cross_cost + (~band).float() * (self.dtw_band_penalty * self._warmup_scale(self.dtw_band_warmup_steps))

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
        if text_mask.shape[1] <= 1:
            return torch.zeros(text_mask.shape[0], target_len, dtype=torch.bool, device=device)
        shifted = text_mask[:, :target_len]
        if shifted.shape[1] < target_len:
            pad = torch.zeros(shifted.shape[0], target_len - shifted.shape[1], dtype=torch.bool, device=device)
            shifted = torch.cat([shifted, pad], dim=1)
        return shifted

    def _update_dtw_gamma(self) -> None:
        if self.dtw is None or self.dtw_gamma_steps <= 0:
            return
        progress = min(1.0, float(self._global_step + 1) / float(self.dtw_gamma_steps))
        self.dtw.gamma = self.dtw_gamma_start + (self.dtw_gamma_end - self.dtw_gamma_start) * progress

    def _warmup_scale(self, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._global_step + 1) / float(warmup_steps))
