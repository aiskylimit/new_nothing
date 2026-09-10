from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.criterions.etp import ETP
from src.criterions.various_divergence import VariousDivergence


def get_hidden_states(outputs) -> Tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("MCW-KD requires model outputs with hidden_states.")
    return tuple(hidden_states)


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
    raise RuntimeError("Could not find input embeddings for MCW-KD.")


def get_output_head(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_output_embeddings"):
        head = encoder.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(encoder, "lm_head"):
        return encoder.lm_head
    raise RuntimeError("Could not find output embedding/lm_head for MCW-KD.")


def project(projector: nn.Module, value: torch.Tensor) -> torch.Tensor:
    target_dtype = next(projector.parameters()).dtype
    return projector(value.to(dtype=target_dtype)).to(dtype=torch.float32)


def require_projector(projectors: Any, name: str) -> nn.Module:
    if not isinstance(projectors, nn.ModuleDict) or name not in projectors:
        raise RuntimeError(
            f"MCW-KD requires a named projector `{name}` in projector_config_path. "
            "Expected projectors: query, s2t, t2s. Optional: ot."
        )
    return projectors[name]


def safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.float().std().clamp_min(1e-6)


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) == 0:
        return len(right)
    if len(right) == 0:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def dist_fn_edit(left: str, right: str) -> int:
    return edit_distance(left, right)


def dtw_alignment(
    series_1: List[str],
    series_2: List[str],
    edit_cache: Dict[Tuple[str, str], int] | None = None,
) -> List[Tuple[int, int]]:
    if len(series_1) == 0 or len(series_2) == 0:
        return []

    # DTW only needs the previous row while filling the table.  Keep one byte
    # per cell for backtracking instead of a Python float/object matrix, which
    # is especially expensive for max-length responses.
    rows, cols = len(series_1), len(series_2)
    previous = [float("inf")] * (cols + 1)
    previous[0] = 0.0
    directions = bytearray(rows * cols)  # 0: diagonal, 1: up, 2: left
    cache = edit_cache if edit_cache is not None else {}

    for i, value_1 in enumerate(series_1):
        current = [float("inf")] * (cols + 1)
        for j, value_2 in enumerate(series_2):
            # Levenshtein distance is symmetric.  A canonical key lets the
            # reverse DTW pass reuse all token-pair distances as well.
            key = (value_1, value_2) if value_1 <= value_2 else (value_2, value_1)
            cost = cache.get(key)
            if cost is None:
                cost = dist_fn_edit(value_1, value_2)
                cache[key] = cost

            up = previous[j + 1]
            left = current[j]
            diagonal = previous[j]
            current[j + 1] = float(cost) + min(up, left, diagonal)

            # Match the original backtracking tie order: diagonal, up, left.
            if i == 0 and j == 0:
                direction = 0
            elif i == 0:
                direction = 2
            elif j == 0:
                direction = 1
            elif diagonal <= up and diagonal <= left:
                direction = 0
            elif up <= left:
                direction = 1
            else:
                direction = 2
            directions[i * cols + j] = direction
        previous = current

    i, j = rows - 1, cols - 1
    aligned: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        aligned.append((i, j))
        move = directions[i * cols + j]
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    aligned.append((0, 0))
    return aligned


def _optional_projector(projectors: Any, name: str) -> nn.Module | None:
    if isinstance(projectors, nn.ModuleDict) and name in projectors:
        return projectors[name]
    return None


def _crop_pair(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = min(left.shape[-1], right.shape[-1])
    return left[..., :dim], right[..., :dim]


def _pairwise_kl_cost_block(teacher_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
    """Original MCW pairwise KL expression for one teacher-token block."""
    return (
        teacher_probs.unsqueeze(1)
        * torch.log(
            (
                teacher_probs.unsqueeze(1)
                / student_probs.unsqueeze(0).clamp_min(1e-9)
            ).clamp_min(1e-9)
        )
    ).sum(dim=-1)


def pairwise_kl_cost(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
    teacher_chunk: int = 32,
) -> torch.Tensor:
    """Exact pairwise KL with bounded forward/backward temporary memory."""
    teacher_probs = teacher_probs.float()
    student_probs = student_probs.float()
    blocks = []
    use_checkpoint = torch.is_grad_enabled() and (teacher_probs.requires_grad or student_probs.requires_grad)
    for start in range(0, teacher_probs.shape[0], teacher_chunk):
        teacher_block = teacher_probs[start : start + teacher_chunk]
        if use_checkpoint:
            block = checkpoint(
                _pairwise_kl_cost_block,
                teacher_block,
                student_probs,
                use_reentrant=False,
            )
        else:
            block = _pairwise_kl_cost_block(teacher_block, student_probs)
        blocks.append(block)
    return torch.cat(blocks, dim=0)


def pairwise_cosine_cost(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine cost without broadcasting the hidden dimension."""
    teacher = F.normalize(teacher.float(), dim=-1, eps=1e-8)
    student = F.normalize(student.float(), dim=-1, eps=1e-8)
    return 1.0 - teacher.matmul(student.transpose(0, 1))


def context_window_mean(seq: torch.Tensor, window: int) -> torch.Tensor:
    """Vectorized variable-width mean used at sequence boundaries."""
    if seq.shape[0] == 0:
        return seq
    positions = torch.arange(seq.shape[0], device=seq.device)
    starts = (positions - window).clamp_min(0)
    ends = (positions + window + 1).clamp_max(seq.shape[0])
    accumulation_dtype = torch.float32 if seq.dtype in (torch.float16, torch.bfloat16) else seq.dtype
    prefix = torch.cat(
        [
            torch.zeros((1, seq.shape[1]), dtype=accumulation_dtype, device=seq.device),
            seq.cumsum(dim=0, dtype=accumulation_dtype),
        ],
        dim=0,
    )
    counts = (ends - starts).to(dtype=accumulation_dtype).unsqueeze(-1)
    means = (prefix.index_select(0, ends) - prefix.index_select(0, starts)) / counts
    return means.to(dtype=seq.dtype)


def topk_values_chunked(logits: torch.Tensor, k: int, sequence_chunk: int = 128) -> torch.Tensor:
    """Top-k in FP32 without copying the entire sequence-vocabulary tensor."""
    chunks = [
        torch.topk(logits[:, start : start + sequence_chunk].float(), k, dim=-1).values
        for start in range(0, logits.shape[1], sequence_chunk)
    ]
    return torch.cat(chunks, dim=1)


def max_softmax_probability_sum(
    logits: torch.Tensor,
    mask: torch.Tensor,
    sequence_chunk: int = 128,
) -> torch.Tensor:
    """Sum masked max probabilities while bounding the temporary FP32 logits."""
    total = logits.new_zeros((), dtype=torch.float32)
    detached = logits.detach()
    for start in range(0, logits.shape[1], sequence_chunk):
        chunk = detached[:, start : start + sequence_chunk].float()
        max_probability = torch.exp(chunk.amax(dim=-1) - torch.logsumexp(chunk, dim=-1))
        total = total + (max_probability * mask[:, start : start + sequence_chunk].float()).sum()
    return total


class MCWKDCriterion(VariousDivergence):
    """
    MCW-KD adapted for the VLM distillation API.

    The reference implementation was written for text-only LLMs. This port uses
    the VLM output ``text_feature_mask`` and label masks so every KD/OT term is
    restricted to text response positions; vision tokens are never selected for
    the loss.
    """

    def __init__(self, args):
        super().__init__(args)
        self.only_save_projector = bool(getattr(args, "only_save_projector", False))
        self.top_k_vocab = int((getattr(args, "top_k_vocab", None) or getattr(args, "mcw_top_k_vocab", 128)) or 128)
        self.tau_seq = float(getattr(args, "mcw_tau_seq", 1.0))
        self.window_size = int(getattr(args, "mcw_window_size", 4) or 4)
        self.total_steps = int(getattr(args, "mcw_total_steps", getattr(args, "max_steps", 0)) or 0)
        self.ot_logits_rate = float(getattr(args, "mcw_ot_logits_rate", 1.0))
        self.ot_hidden_rate = float(getattr(args, "mcw_ot_hidden_rate", 1.0))
        self._global_step = 0
        self.etp = ETP(
            sinkhorn_alpha=float(getattr(args, "mcw_sinkhorn_alpha", 0.1)),
            max_iter=int(getattr(args, "mcw_sinkhorn_iter", 100) or 100),
        )
        self.register_buffer("cost_weights_logits", torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32))
        self.register_buffer("cost_weights_hidden", torch.tensor([0.3, 0.5, 0.2], dtype=torch.float32))

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running MCW-KD.")

        student_outputs = distiller.student(**student_inputs)
        labels = student_inputs["labels"].to(device=student_outputs.logits.device)
        supervised_loss = self.compute_cross_entropy_loss(student_outputs.logits, labels)

        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        teacher_labels, _teacher_mask = self.teacher_targets(teacher_inputs, student_outputs.logits.device)
        masks = self._shifted_targets_and_masks(
            student_inputs,
            teacher_inputs,
            student_outputs,
            teacher_outputs,
            labels,
            teacher_labels,
        )
        (
            student_input,
            target,
            student_mask,
            teacher_input,
            teacher_target,
            teacher_mask,
        ) = masks

        kd_loss, extra = self._dual_space_kd_loss(
            distiller,
            student_outputs,
            teacher_outputs,
            student_input,
            target,
            student_mask,
            teacher_input,
            teacher_target,
            teacher_mask,
        )
        ot_logits_loss, ot_logits_extra = self._ot_logits_loss(
            distiller,
            student_outputs,
            teacher_outputs,
            student_mask,
            teacher_mask,
            target.shape[1],
            teacher_target.shape[1],
        )
        ot_hidden_loss, ot_hidden_extra = self._ot_hidden_loss(
            distiller,
            student_inputs,
            teacher_inputs,
            student_outputs,
            teacher_outputs,
            student_mask,
            teacher_mask,
            target.shape[1],
            teacher_target.shape[1],
        )

        loss = (
            supervised_loss
            + self.kd_rate * kd_loss
            + self.ot_logits_rate * ot_logits_loss
            + self.ot_hidden_rate * ot_hidden_loss
        )
        self._global_step += 1

        result = {
            "loss": loss,
            "supervised_loss": supervised_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "mcw_ot_logits_loss": ot_logits_loss.detach(),
            "mcw_ot_hidden_loss": ot_hidden_loss.detach(),
            "token_accuracy": self.compute_token_accuracy(student_outputs.logits, labels).detach(),
            "token_correct": self.compute_token_correct(student_outputs.logits, labels).detach(),
            "token_count": self.compute_token_count(student_outputs.logits, labels).detach(),
        }
        result.update({name: value.detach() for name, value in extra.items()})
        result.update({name: value.detach() for name, value in ot_logits_extra.items()})
        result.update({name: value.detach() for name, value in ot_hidden_extra.items()})
        return result

    def _shifted_targets_and_masks(
        self,
        student_inputs: Dict[str, torch.Tensor],
        teacher_inputs: Dict[str, torch.Tensor],
        student_outputs,
        teacher_outputs,
        labels: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        student_input, target, student_mask = self.shift_inputs_for_causal_targets(
            student_inputs["input_ids"].to(device=labels.device),
            labels,
        )
        teacher_input, teacher_target, teacher_mask = self.shift_inputs_for_causal_targets(
            teacher_inputs["input_ids"].to(device=labels.device),
            teacher_labels.to(device=labels.device),
        )
        student_mask = student_mask & self._shifted_text_mask(student_outputs, target.shape[1], labels.device)
        teacher_mask = teacher_mask & self._shifted_text_mask(teacher_outputs, teacher_target.shape[1], labels.device)
        target = target.masked_fill(~student_mask, self.padding_id)
        teacher_target = teacher_target.masked_fill(~teacher_mask, self.padding_id)
        return student_input, target, student_mask, teacher_input, teacher_target, teacher_mask

    def _dual_space_kd_loss(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_input: torch.Tensor,
        target: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_target: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_hidden = get_hidden_states(student_outputs)[-1][:, : target.shape[1]]
        teacher_hidden = get_hidden_states(teacher_outputs)[-1][:, : teacher_target.shape[1]].to(device=student_hidden.device)
        student_logits = student_outputs.logits[:, : target.shape[1]]
        teacher_logits = teacher_outputs.logits[:, : teacher_target.shape[1]].to(device=student_hidden.device)

        student_embed = get_input_embeddings(distiller.student)
        teacher_embed = get_input_embeddings(distiller.teacher)

        formal_target = torch.where(student_mask, target, torch.zeros_like(target))
        formal_student_input = torch.where(student_mask, student_input, torch.zeros_like(student_input))
        formal_teacher_target = torch.where(teacher_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_mask, teacher_input, torch.zeros_like(teacher_input))

        student_input_embeds = student_embed(formal_student_input).detach()
        student_target_embeds = student_embed(formal_target).detach()
        teacher_input_embeds = teacher_embed(formal_teacher_input).detach().to(device=student_hidden.device)
        teacher_target_embeds = teacher_embed(formal_teacher_target).detach().to(device=student_hidden.device)

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
        t2s_hidden = torch.matmul(t2s_weight, teacher_value).to(dtype=student_hidden.dtype)
        t2s_logits = get_output_head(distiller.student)(t2s_hidden)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target, shift=False)

        t2s_acc_mask = t2s_logits.argmax(dim=-1).eq(target) & student_mask
        t2s_acc = t2s_acc_mask.sum()
        max_t2s_prob = max_softmax_probability_sum(t2s_logits, student_mask)

        if self.only_save_projector:
            return t2s_ce_loss, {
                "t2s_ce_loss": t2s_ce_loss,
                "t2s_acc": t2s_acc,
                "max_t2s_prob": max_t2s_prob,
            }

        t2s_kd_vec = self.dist_func(
            student_logits,
            t2s_logits.detach(),
            target,
            teacher_temperature=self.teacher_temperature,
            reduction="none",
        )
        t2s_gate = student_mask.float() * t2s_acc_mask.float()
        t2s_kd_loss = (t2s_kd_vec * t2s_gate).sum()

        s2t_weight = torch.softmax(align.transpose(-1, -2), dim=-1)
        s2t_hidden = torch.matmul(s2t_weight, student_value).to(dtype=teacher_hidden.dtype)
        s2t_logits = get_output_head(distiller.teacher)(s2t_hidden)
        s2t_kd_vec = self.compute_forward_kl_divergence(
            s2t_logits,
            teacher_logits.to(device=s2t_logits.device),
            teacher_target,
            reduction="none",
        )
        s2t_kd_loss = (s2t_kd_vec * teacher_mask.float()).sum()
        s2t_acc = (s2t_logits.argmax(dim=-1).eq(teacher_target) * teacher_mask).sum()
        kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
        return kd_loss, {
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "s2t_kd_loss": s2t_kd_loss,
            "t2s_acc": t2s_acc,
            "s2t_acc": s2t_acc,
            "max_t2s_prob": max_t2s_prob,
        }

    def _ot_logits_loss(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        student_len: int,
        teacher_len: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_logits = student_outputs.logits[:, :student_len]
        teacher_logits = teacher_outputs.logits[:, :teacher_len].to(device=student_logits.device)
        student_hidden = get_hidden_states(student_outputs)[-1][:, :student_len]
        teacher_hidden = get_hidden_states(teacher_outputs)[-1][:, :teacher_len].to(device=student_hidden.device)
        teacher_hidden = self._project_teacher_hidden(distiller, teacher_hidden, student_hidden.shape[-1])
        student_hidden, teacher_hidden = _crop_pair(student_hidden, teacher_hidden)

        k = max(1, min(self.top_k_vocab, student_logits.shape[-1], teacher_logits.shape[-1]))
        student_topk = topk_values_chunked(student_logits, k)
        teacher_topk = topk_values_chunked(teacher_logits, k)
        tau = max(self.tau_seq, 1e-6)
        frac = 1.0 if self.total_steps <= 0 else min(float(self._global_step + 1) / float(self.total_steps), 1.0)
        interpolation_t = 0.1 + 0.9 * frac
        if student_topk.shape == teacher_topk.shape:
            interp_teacher = (1.0 - interpolation_t) * student_topk + interpolation_t * teacher_topk
        else:
            interp_teacher = teacher_topk

        student_probs = F.softmax(student_topk / tau, dim=-1)
        teacher_probs = F.softmax(interp_teacher / tau, dim=-1)
        total = student_logits.new_zeros(())
        stats = {"mcw_avg_ot_logit_l2": [], "mcw_avg_ot_logit_kl": [], "mcw_avg_ot_logit_salience": []}
        batches = 0

        for index in range(student_logits.shape[0]):
            s_pos = student_mask[index].nonzero(as_tuple=False).flatten()
            t_pos = teacher_mask[index].nonzero(as_tuple=False).flatten()
            if s_pos.numel() == 0 or t_pos.numel() == 0:
                continue
            batches += 1
            sp = student_probs[index].index_select(0, s_pos)
            tp = teacher_probs[index].index_select(0, t_pos)
            c_l2 = torch.cdist(tp, sp, p=2)
            c_kl = pairwise_kl_cost(tp, sp)
            s_seq = student_hidden[index].index_select(0, s_pos)
            t_seq = teacher_hidden[index].index_select(0, t_pos)
            sal_s, sal_t = self._salience_scores(distiller, s_seq, t_seq)
            c_sal = torch.abs(sal_t.unsqueeze(1) - sal_s.unsqueeze(0))
            total = total + self.etp(self._weighted_cost([c_l2, c_kl, c_sal], self.cost_weights_logits, student_logits.device))[0].float()
            stats["mcw_avg_ot_logit_l2"].append(c_l2.mean())
            stats["mcw_avg_ot_logit_kl"].append(c_kl.mean())
            stats["mcw_avg_ot_logit_salience"].append(c_sal.mean())

        if batches == 0:
            return student_logits.new_zeros(()), {}
        return total / batches, {name: torch.stack(values).mean() for name, values in stats.items() if values}

    def _ot_hidden_loss(
        self,
        distiller: Any,
        student_inputs: Dict[str, torch.Tensor],
        teacher_inputs: Dict[str, torch.Tensor],
        student_outputs,
        teacher_outputs,
        student_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        student_len: int,
        teacher_len: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_hidden = get_hidden_states(student_outputs)[-1][:, :student_len]
        teacher_hidden = get_hidden_states(teacher_outputs)[-1][:, :teacher_len].to(device=student_hidden.device)
        teacher_hidden = self._project_teacher_hidden(distiller, teacher_hidden, student_hidden.shape[-1])
        student_hidden, teacher_hidden = _crop_pair(student_hidden, teacher_hidden)
        total = student_hidden.new_zeros(())
        stats = {"mcw_avg_ot_hidden_edit": [], "mcw_avg_ot_hidden_l2": [], "mcw_avg_ot_hidden_cos": []}
        batches = 0

        for index in range(student_hidden.shape[0]):
            s_pos = student_mask[index].nonzero(as_tuple=False).flatten()
            t_pos = teacher_mask[index].nonzero(as_tuple=False).flatten()
            if s_pos.numel() == 0 or t_pos.numel() == 0:
                continue
            batches += 1
            s_seq = student_hidden[index].index_select(0, s_pos)
            t_seq = teacher_hidden[index].index_select(0, t_pos)
            ctx_s = self._context_repr(s_seq, self.window_size)
            ctx_t = self._context_repr(t_seq, self.window_size)
            student_tokens = self._tokens_for_positions(
                distiller,
                student_inputs["input_ids"].to(device=student_mask.device),
                s_pos,
                index,
                teacher=False,
            )
            teacher_tokens = self._tokens_for_positions(
                distiller,
                teacher_inputs["input_ids"].to(device=teacher_mask.device),
                t_pos,
                index,
                teacher=True,
            )
            c_edit = self._dtw_edit_cost(student_tokens, teacher_tokens, t_pos.numel(), s_pos.numel(), student_hidden.device)
            c_l2 = torch.cdist(ctx_t.float(), ctx_s.float(), p=2)
            c_l2 = c_l2 / c_l2.detach().amax().clamp_min(1e-6)
            c_cos = pairwise_cosine_cost(ctx_t, ctx_s)
            total = total + self.etp(self._weighted_cost([c_edit, c_l2, c_cos], self.cost_weights_hidden, student_hidden.device))[0].float()
            stats["mcw_avg_ot_hidden_edit"].append(c_edit.mean())
            stats["mcw_avg_ot_hidden_l2"].append(c_l2.mean())
            stats["mcw_avg_ot_hidden_cos"].append(c_cos.mean())

        if batches == 0:
            return student_hidden.new_zeros(()), {}
        return total / batches, {name: torch.stack(values).mean() for name, values in stats.items() if values}

    def _salience_scores(
        self,
        distiller: Any,
        student_seq: torch.Tensor,
        teacher_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projector = _optional_projector(distiller.projectors, "salience")
        if projector is not None:
            sal_s = torch.sigmoid(project(projector, student_seq)).squeeze(-1)
            sal_t = torch.sigmoid(project(projector, teacher_seq)).squeeze(-1)
            return sal_s.float(), sal_t.float()

        sal_s = torch.linalg.vector_norm(student_seq.float(), dim=-1)
        sal_t = torch.linalg.vector_norm(teacher_seq.float(), dim=-1)
        scale = torch.cat([sal_s, sal_t]).detach().amax().clamp_min(1e-6)
        return sal_s / scale, sal_t / scale

    def _project_teacher_hidden(self, distiller: Any, teacher_hidden: torch.Tensor, student_dim: int) -> torch.Tensor:
        projector = _optional_projector(distiller.projectors, "ot") or _optional_projector(distiller.projectors, "t2s")
        if projector is not None:
            try:
                return project(projector, teacher_hidden).to(device=teacher_hidden.device)
            except RuntimeError:
                pass
        if teacher_hidden.shape[-1] == student_dim:
            return teacher_hidden
        dim = min(teacher_hidden.shape[-1], student_dim)
        return teacher_hidden[..., :dim]

    def _weighted_cost(self, costs: list[torch.Tensor], weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        weights = weights.to(device=device, dtype=torch.float32)
        weighted = sum(weight * cost.float() for weight, cost in zip(weights, costs))
        return weighted.masked_fill(weighted.isnan() | weighted.isinf(), 0.0)

    def _context_repr(self, seq: torch.Tensor, window: int) -> torch.Tensor:
        return context_window_mean(seq, window)

    def _dtw_edit_cost(
        self,
        student_tokens: List[str],
        teacher_tokens: List[str],
        teacher_len: int,
        student_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        edit_cache: Dict[Tuple[str, str], int] = {}
        # DTW itself runs on CPU.  Populate its sparse path there as well and
        # perform one device transfer instead of thousands of scalar CUDA writes.
        c_s2t = torch.zeros((student_len, teacher_len), dtype=torch.float32)
        for i, j in dtw_alignment(student_tokens, teacher_tokens, edit_cache):
            if i < student_len and j < teacher_len:
                left, right = student_tokens[i], teacher_tokens[j]
                key = (left, right) if left <= right else (right, left)
                c_s2t[i, j] = float(edit_cache[key])

        c_t2s = torch.zeros((teacher_len, student_len), dtype=torch.float32)
        for j, i in dtw_alignment(teacher_tokens, student_tokens, edit_cache):
            if j < teacher_len and i < student_len:
                left, right = teacher_tokens[j], student_tokens[i]
                key = (left, right) if left <= right else (right, left)
                c_t2s[j, i] = float(edit_cache[key])

        return ((c_s2t.t() + c_t2s) / 2.0).to(device=device)

    def _tokens_for_positions(
        self,
        distiller: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_index: int,
        teacher: bool,
    ) -> List[str]:
        ids = input_ids[batch_index].index_select(0, positions.to(device=input_ids.device)).detach().cpu().tolist()
        tokenizer = self._tokenizer(distiller, teacher=teacher)
        if tokenizer is None:
            return [str(token_id) for token_id in ids]
        try:
            tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(ids)
        if tokens is None:
            return [str(token_id) for token_id in ids]
        if isinstance(tokens, str):
            tokens = [tokens]
        tokens = [str(token) for token in tokens]
        if len(tokens) < len(ids):
            tokens.extend(str(token_id) for token_id in ids[len(tokens):])
        return tokens[: len(ids)]

    def _tokenizer(self, distiller: Any, teacher: bool):
        processor_getter = distiller.get_teacher_processor if teacher else distiller.get_student_processor
        try:
            processor = processor_getter()
        except Exception:
            return None
        return getattr(processor, "tokenizer", processor)

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
