from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.utils import print_rank


IGNORE_INDEX = -100


def _last_hidden(outputs) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("EM-KD requires model outputs with hidden_states.")
    return hidden_states[-1]


def _get_output_head(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_output_embeddings"):
        head = encoder.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(encoder, "lm_head"):
        return encoder.lm_head
    raise RuntimeError("Could not find output embedding/lm_head for EM-KD.")


def _crop_last_dim(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = min(left.shape[-1], right.shape[-1])
    return left[..., :dim], right[..., :dim]


def _crop_seq_and_last_dim(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_len = min(left.shape[1], right.shape[1])
    logit_dim = min(left.shape[-1], right.shape[-1])
    return left[:, -seq_len:, :logit_dim], right[:, -seq_len:, :logit_dim], seq_len


def _reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_logits, teacher_logits = _crop_last_dim(student_logits, teacher_logits)
    temperature = max(float(temperature), 1e-6)
    log_p_s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    p_s = log_p_s.exp()
    return (p_s * (log_p_s - log_p_t)).sum(dim=-1).mean() * (temperature ** 2)


def _hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = cost.device
    idx_s, idx_t = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(idx_s, dtype=torch.long, device=device),
        torch.as_tensor(idx_t, dtype=torch.long, device=device),
    )


def _mask_for_sample(mask: Optional[torch.Tensor], index: int, hidden: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    return mask[index].to(device=hidden.device, dtype=torch.bool)


def _limit_tokens(hidden: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens is None or max_tokens <= 0 or hidden.shape[0] <= max_tokens:
        return hidden
    idx = torch.linspace(
        0,
        hidden.shape[0] - 1,
        steps=max_tokens,
        device=hidden.device,
    ).round().long()
    return hidden.index_select(0, idx)


def _matched_response_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_logits, teacher_logits, min_seq_len = _crop_seq_and_last_dim(student_logits, teacher_logits)

    if labels is not None:
        response_mask = labels[:, -min_seq_len:].to(student_logits.device).ne(IGNORE_INDEX)
    elif attention_mask is not None:
        response_mask = attention_mask[:, -min_seq_len:].to(student_logits.device).bool()
    else:
        response_mask = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)

    return student_logits[response_mask], teacher_logits[response_mask], response_mask


class EMKDCriterion(nn.Module):
    """
    EM-KD for generative VLM training.

    The loss combines supervised assistant-token CE, reverse-KL response-logit
    distillation, matched vision-logit distillation, and vision-language
    affinity distillation. Vision/text token groups are selected with the
    backbone-provided ``vision_feature_mask`` and ``text_feature_mask``.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = float(getattr(args, "em_kd_alpha", 0.5))
        self.beta = float(getattr(args, "em_kd_beta", 0.25))
        self.gamma = float(getattr(args, "em_kd_gamma", 25.0))
        self.temperature = float(getattr(args, "em_kd_temperature", 1.0))
        self.max_vision_tokens = int(getattr(args, "em_kd_max_vision_tokens", 512) or 0)
        self.max_text_tokens = int(getattr(args, "em_kd_max_text_tokens", 1024) or 0)

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running EM-KD.")

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
        supervised_loss = student_outputs.loss
        if supervised_loss is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        response_loss = self._response_logit_distillation(student_outputs, teacher_outputs, student_inputs)
        vision_loss, vision_logit_loss, affinity_loss, matched_count = self._vision_distillation(
            distiller, student_outputs, teacher_outputs
        )

        total_loss = self.alpha * supervised_loss + (1.0 - self.alpha) * response_loss + vision_loss
        return {
            "loss": total_loss,
            "supervised_loss": supervised_loss.detach(),
            "response_kd_loss": response_loss.detach(),
            "vision_kd_loss": vision_loss.detach(),
            "vision_logit_loss": vision_logit_loss.detach(),
            "affinity_loss": affinity_loss.detach(),
            "matched_vision_samples": supervised_loss.new_tensor(float(matched_count)),
        }

    def _response_logit_distillation(self, student_outputs, teacher_outputs, student_inputs) -> torch.Tensor:
        student_logits = getattr(student_outputs, "logits", None)
        teacher_logits = getattr(teacher_outputs, "logits", None)
        if student_logits is None or teacher_logits is None:
            raise RuntimeError("EM-KD response distillation requires student and teacher logits.")

        student_response, teacher_response, response_mask = _matched_response_logits(
            student_logits,
            teacher_logits,
            student_inputs.get("labels"),
            student_inputs.get("attention_mask"),
        )
        if response_mask.sum() == 0:
            return student_logits.new_zeros(())
        return _reverse_kl(student_response, teacher_response.to(student_response.device), self.temperature)

    def _vision_distillation(self, distiller: Any, student_outputs, teacher_outputs):
        student_hidden = _last_hidden(student_outputs)
        teacher_hidden = _last_hidden(teacher_outputs)
        student_head = _get_output_head(distiller.student)
        teacher_head = _get_output_head(distiller.teacher)

        student_vision_mask = getattr(student_outputs, "vision_feature_mask", None)
        teacher_vision_mask = getattr(teacher_outputs, "vision_feature_mask", None)
        student_text_mask = getattr(student_outputs, "text_feature_mask", None)
        teacher_text_mask = getattr(teacher_outputs, "text_feature_mask", None)

        vision_logit_losses = []
        affinity_losses = []
        batch_size = student_hidden.shape[0]

        for i in range(batch_size):
            sv_mask = _mask_for_sample(student_vision_mask, i, student_hidden)
            tv_mask = _mask_for_sample(teacher_vision_mask, i, teacher_hidden)
            st_mask = _mask_for_sample(student_text_mask, i, student_hidden)
            tt_mask = _mask_for_sample(teacher_text_mask, i, teacher_hidden)

            if not (sv_mask.any() and tv_mask.any() and st_mask.any() and tt_mask.any()):
                print_rank(f"Warning: sample {i} has empty vision/text masks; skipping vision KD for this sample.")
                continue

            vhs_s = _limit_tokens(student_hidden[i][sv_mask], self.max_vision_tokens)
            vhs_t = _limit_tokens(teacher_hidden[i][tv_mask], self.max_vision_tokens)
            lhs_s = _limit_tokens(student_hidden[i][st_mask], self.max_text_tokens)
            lhs_t = _limit_tokens(teacher_hidden[i][tt_mask], self.max_text_tokens)

            vl_s, vl_t = _crop_last_dim(student_head(vhs_s), teacher_head(vhs_t))
            with torch.no_grad():
                cost = torch.cdist(vl_s.float(), vl_t.to(vl_s.device).float(), p=1).detach()
                idx_s, idx_t = _hungarian_match(cost)

            vl_s_matched = vl_s[idx_s]
            vl_t_matched = vl_t[idx_t].to(vl_s_matched.device)
            vision_logit_losses.append(_reverse_kl(vl_s_matched, vl_t_matched, self.temperature))

            vhs_s_matched = vhs_s[idx_s]
            vhs_t_matched = vhs_t[idx_t].to(vhs_s_matched.device)
            affinity_s = F.cosine_similarity(vhs_s_matched.unsqueeze(1), lhs_s.unsqueeze(0), dim=-1)
            affinity_t = F.cosine_similarity(vhs_t_matched.unsqueeze(1), lhs_t.unsqueeze(0), dim=-1)
            min_text_len = min(affinity_s.shape[1], affinity_t.shape[1])
            affinity_losses.append(F.smooth_l1_loss(affinity_s[:, :min_text_len], affinity_t[:, :min_text_len]))

        if not vision_logit_losses:
            zero = student_hidden.new_zeros(())
            return zero, zero, zero, 0

        vision_logit_loss = torch.stack(vision_logit_losses).mean()
        affinity_loss = torch.stack(affinity_losses).mean()
        vision_loss = self.beta * vision_logit_loss + self.gamma * affinity_loss
        return vision_loss, vision_logit_loss, affinity_loss, len(vision_logit_losses)
