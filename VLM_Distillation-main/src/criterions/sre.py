from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100
DEFAULT_HIDDEN_LOSS_WEIGHTS = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10)


def _as_layer_list(value: Optional[Sequence[int]]) -> list[int]:
    return list(value or [])


def _last_hidden(outputs) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("SRE requires model outputs with hidden_states.")
    return hidden_states[-1]


def _crop_seq(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_len = min(left.shape[1], right.shape[1])
    return left[:, -seq_len:], right[:, -seq_len:], seq_len


def _crop_last_dim(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = min(left.shape[-1], right.shape[-1])
    return left[..., :dim], right[..., :dim]


def _crop_seq_and_last_dim(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    left, right, seq_len = _crop_seq(left, right)
    left, right = _crop_last_dim(left, right)
    return left, right, seq_len


def _crop_seq_only(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_len = min(left.shape[1], right.shape[1])
    return left[:, -seq_len:], right[:, -seq_len:], seq_len


def _get_output_head(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_output_embeddings"):
        head = encoder.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(encoder, "lm_head"):
        return encoder.lm_head
    raise RuntimeError("Could not find output embedding/lm_head for SRE.")


def _pool_hidden_states(hidden: torch.Tensor, safe_idx: torch.Tensor, pooler_mask: torch.Tensor) -> torch.Tensor:
    safe_idx = safe_idx.to(device=hidden.device, dtype=torch.long)
    pooler_mask = pooler_mask.to(device=hidden.device, dtype=hidden.dtype)
    safe_idx = safe_idx.clamp(min=0, max=hidden.shape[1] - 1)

    batch_size, n_span, span_width = safe_idx.shape
    batch_idx = torch.arange(batch_size, device=hidden.device)[:, None, None].expand(batch_size, n_span, span_width)
    gathered = hidden[batch_idx, safe_idx]
    gathered = gathered * pooler_mask.unsqueeze(-1)
    denom = pooler_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return gathered.sum(dim=2) / denom


def _span_weights(pooler_mask: torch.Tensor, p: float) -> torch.Tensor:
    weights = pooler_mask.to(dtype=torch.float32).sum(dim=-1)
    weights = weights.clamp(max=1.0).pow(max(float(p), 1e-5))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)


def _normalize_weights(weights: torch.Tensor, p: float) -> torch.Tensor:
    weights = weights.to(dtype=torch.float32).clamp_min(0.0).pow(max(float(p), 1e-5))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)


def _token_weights(
    student_outputs,
    student_inputs: Dict[str, Any],
    seq_len: int,
    p: float,
) -> torch.Tensor:
    labels = student_inputs.get("labels")
    if labels is not None:
        mask = labels[:, -seq_len:].ne(IGNORE_INDEX)
    else:
        text_mask = getattr(student_outputs, "text_feature_mask", None)
        if text_mask is not None:
            mask = text_mask[:, -seq_len:].bool()
        elif student_inputs.get("attention_mask") is not None:
            mask = student_inputs["attention_mask"][:, -seq_len:].bool()
        else:
            device = _last_hidden(student_outputs).device
            mask = torch.ones((student_inputs["input_ids"].shape[0], seq_len), dtype=torch.bool, device=device)

    weights = mask.to(dtype=torch.float32)
    weights = weights.pow(max(float(p), 1e-5))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)


def _pair_weights(token_weights: torch.Tensor) -> torch.Tensor:
    weights = token_weights.unsqueeze(2) * token_weights.unsqueeze(1)
    seq_len = weights.shape[-1]
    eye = torch.eye(seq_len, dtype=torch.bool, device=weights.device)
    weights = weights.masked_fill(eye.unsqueeze(0), 0.0)
    return weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-5)


def _layer_attention(outputs, hidden_layer: int):
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        return None

    attn_idx = max(int(hidden_layer) - 1, 0)
    try:
        if attn_idx >= len(attentions):
            return None
        return attentions[attn_idx]
    except TypeError:
        if attentions.ndim < 4 or attn_idx >= attentions.shape[0]:
            return None
        return attentions[attn_idx]


def _causal_attention_weights(attention: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], seq_len: int):
    if attention is None or attention.ndim != 4:
        return None
    if attention.shape[-1] != seq_len:
        return None

    weights = attention.sum(dim=1)[:, -1].detach().float()
    if attention_mask is not None and attention_mask.shape[-1] == seq_len:
        weights = weights * attention_mask.to(device=weights.device, dtype=weights.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)


def _pool_hidden_states_with_weights(
    hidden: torch.Tensor,
    safe_idx: torch.Tensor,
    pooler_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    attention: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    safe_idx = safe_idx.to(device=hidden.device, dtype=torch.long)
    pooler_mask = pooler_mask.to(device=hidden.device, dtype=hidden.dtype)
    safe_idx = safe_idx.clamp(min=0, max=hidden.shape[1] - 1)

    batch_size, n_span, span_width = safe_idx.shape
    batch_idx = torch.arange(batch_size, device=hidden.device)[:, None, None].expand(batch_size, n_span, span_width)
    gathered = hidden[batch_idx, safe_idx] * pooler_mask.unsqueeze(-1)

    token_weights = _causal_attention_weights(attention, attention_mask, hidden.shape[1])
    if token_weights is None:
        denom = pooler_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = gathered.sum(dim=2) / denom
        span_weight = pooler_mask.sum(dim=-1).clamp(max=1.0)
        return pooled, span_weight

    token_weights = token_weights.to(device=hidden.device, dtype=hidden.dtype)
    span_token_weights = token_weights[batch_idx, safe_idx] * pooler_mask
    pooled = (gathered * span_token_weights.unsqueeze(-1)).sum(dim=2) / span_token_weights.sum(
        dim=2, keepdim=True
    ).clamp_min(1e-5)
    return pooled, span_token_weights.sum(dim=2)


def _align_hidden_dims(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projector: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bring student and teacher hidden states to a common dimension.

    With a projector, student is linearly mapped into teacher_dim; this is the
    principled path used by the paper. Without one, fall back to cropping the
    larger side to ``min(student_dim, teacher_dim)`` for backwards compat with
    legacy configs (kept only as a safety net; not recommended for new runs).
    """
    if projector is not None:
        target_dtype = next(projector.parameters()).dtype
        student_hidden = projector(student_hidden.to(target_dtype)).to(torch.float32)
        teacher_hidden = teacher_hidden.to(device=student_hidden.device, dtype=torch.float32)
        return student_hidden, teacher_hidden
    return _crop_last_dim(student_hidden, teacher_hidden)


def _weighted_cosine_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weights: torch.Tensor,
    projector: Optional[nn.Module] = None,
) -> torch.Tensor:
    student_hidden, teacher_hidden = _align_hidden_dims(student_hidden, teacher_hidden, projector)
    student_hidden = F.normalize(student_hidden.float(), dim=-1, eps=1e-5)
    teacher_hidden = F.normalize(teacher_hidden.float(), dim=-1, eps=1e-5)
    per_token = 1.0 - (student_hidden * teacher_hidden).sum(dim=-1)
    return (per_token * weights.to(per_token.device)).sum() / student_hidden.shape[0]


def _geometry_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    pair_weights: torch.Tensor,
    projector: Optional[nn.Module] = None,
) -> torch.Tensor:
    student_hidden, teacher_hidden = _align_hidden_dims(student_hidden, teacher_hidden, projector)
    student_hidden = F.normalize(student_hidden.float(), dim=-1, eps=1e-5)
    teacher_hidden = F.normalize(teacher_hidden.float(), dim=-1, eps=1e-5)
    student_scores = torch.matmul(student_hidden, student_hidden.transpose(-1, -2))
    teacher_scores = torch.matmul(teacher_hidden, teacher_hidden.transpose(-1, -2))
    loss = F.mse_loss(student_scores, teacher_scores, reduction="none")
    return (loss * pair_weights.to(loss.device)).sum() / student_hidden.shape[0]


def _soft_label_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_logits, teacher_logits = _crop_last_dim(student_logits, teacher_logits)
    temperature = max(float(temperature), 1e-6)
    log_probs_s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    probs_t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(log_probs_s, probs_t, reduction="none").sum(dim=-1)
    valid_mask = valid_mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * valid_mask).sum() / student_logits.shape[0]


class SRECriterion(nn.Module):
    """
    SRE-style distillation for the generative VLM_Distill pipeline.

    This is a self-contained adaptation of the reference SRE code: it uses the
    current batch labels/text masks as token weights instead of external SRE
    pooler indices, and it does not import anything from the temporary ``sre``
    reference folder.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = float(getattr(args, "sre_alpha", 0.5))
        self.p = max(float(getattr(args, "sre_p", 1.0)), 1e-5)
        self.span_loss_weight = float(getattr(args, "sre_span_loss_weight", 1.0))
        self.geom_loss_weight = float(getattr(args, "sre_geom_loss_weight", 50))
        self.logit_loss_weight = float(getattr(args, "sre_logit_loss_weight", 1.0))
        self.temperature = float(getattr(args, "sre_temperature", 2.0))
        self.use_projector = bool(getattr(args, "sre_use_projector", False))
        self._shared_vocab_cpu = None
        self._shared_vocab_cache = {}

    def _get_projector(self, distiller: Any, idx: int) -> Optional[nn.Module]:
        if not self.use_projector:
            return None
        projectors = getattr(distiller, "projectors", None)
        if projectors is None or len(projectors) == 0:
            return None
        # Layer-mapping-aware lookup: idx maps to projector position when mapping is
        # non-empty; otherwise fall back to the single default projector.
        if isinstance(projectors, nn.ModuleList):
            if idx < len(projectors):
                return projectors[idx]
            return projectors[0]
        # nn.ModuleDict (projector_config_path mode) is not supported by SRE
        # directly; caller should configure layer_mapping path.
        return None

    @staticmethod
    def _processor_tokenizer(processor):
        return getattr(processor, "tokenizer", processor)

    def _build_shared_vocab_cpu(self, distiller) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._shared_vocab_cpu is not None:
            return self._shared_vocab_cpu

        student_tokenizer = self._processor_tokenizer(distiller.get_student_processor())
        teacher_tokenizer = self._processor_tokenizer(distiller.get_teacher_processor())
        student_vocab = student_tokenizer.get_vocab()
        teacher_vocab = teacher_tokenizer.get_vocab()

        student_ids = []
        teacher_ids = []
        for token, student_id in student_vocab.items():
            teacher_id = teacher_vocab.get(token)
            if teacher_id is None:
                continue
            student_ids.append(student_id)
            teacher_ids.append(teacher_id)

        if not student_ids:
            raise RuntimeError("SRE soft-label KD found no shared vocabulary between student and teacher tokenizers.")

        self._shared_vocab_cpu = (
            torch.tensor(student_ids, dtype=torch.long),
            torch.tensor(teacher_ids, dtype=torch.long),
        )
        return self._shared_vocab_cpu

    def _shared_vocab_indices(
        self,
        distiller,
        device: torch.device,
        student_vocab_size: int,
        teacher_vocab_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (str(device), student_vocab_size, teacher_vocab_size)
        if cache_key in self._shared_vocab_cache:
            return self._shared_vocab_cache[cache_key]

        student_ids_cpu, teacher_ids_cpu = self._build_shared_vocab_cpu(distiller)
        valid = (student_ids_cpu < student_vocab_size) & (teacher_ids_cpu < teacher_vocab_size)
        if not valid.any():
            raise RuntimeError(
                "SRE soft-label KD shared vocabulary ids are outside the current logits dimensions."
            )

        mapped = (
            student_ids_cpu[valid].to(device=device),
            teacher_ids_cpu[valid].to(device=device),
        )
        self._shared_vocab_cache[cache_key] = mapped
        return mapped

    def _aligned_soft_label_loss(
        self,
        distiller,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        student_ids, teacher_ids = self._shared_vocab_indices(
            distiller,
            student_logits.device,
            student_logits.shape[-1],
            teacher_logits.shape[-1],
        )
        student_logits = student_logits.index_select(-1, student_ids)
        teacher_logits = teacher_logits.to(student_logits.device).index_select(-1, teacher_ids)
        return _soft_label_loss(student_logits, teacher_logits, valid_mask, self.temperature)

    @staticmethod
    def _hidden_layer_weights(n_layers: int, device: torch.device) -> torch.Tensor:
        base = torch.tensor(DEFAULT_HIDDEN_LOSS_WEIGHTS, dtype=torch.float32, device=device)
        if n_layers <= base.numel():
            weights = base[:n_layers]
        else:
            repeat = (n_layers + base.numel() - 1) // base.numel()
            weights = base.repeat(repeat)[:n_layers]
        return weights / weights.sum()

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running SRE.")

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
        hard_loss = student_outputs.loss
        if hard_loss is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        kd_loss, span_loss, geom_loss, logit_loss = self._knowledge_distillation_loss(
            distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs
        )
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss

        return {
            "loss": loss,
            "hard_loss": hard_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "sre_span_loss": span_loss.detach(),
            "sre_geom_loss": geom_loss.detach(),
            "sre_logit_loss": logit_loss.detach(),
        }

    def _knowledge_distillation_loss(self, distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs):
        if self._has_pooler(student_inputs, teacher_inputs):
            return self._span_pooled_distillation_loss(
                distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs
            )

        student_last, teacher_last, seq_len = _crop_seq(_last_hidden(student_outputs), _last_hidden(teacher_outputs))
        token_weights = _token_weights(student_outputs, student_inputs, seq_len, self.p).to(student_last.device)
        pair_weights = _pair_weights(token_weights)

        span_loss = self._span_loss(distiller, student_outputs, teacher_outputs, token_weights, seq_len)
        # Geometry projects student to teacher_dim with the same last-layer
        # projector used by the span loss; falls back to crop when missing.
        geom_projector = self._get_projector(distiller, idx=-1)
        geom_loss = _geometry_loss(
            student_last, teacher_last.to(student_last.device), pair_weights, projector=geom_projector
        )

        student_logits, teacher_logits, _ = _crop_seq_only(student_outputs.logits, teacher_outputs.logits)
        logit_loss = self._aligned_soft_label_loss(distiller, student_logits, teacher_logits, token_weights.gt(0))

        kd_loss = (
            self.span_loss_weight * span_loss
            + self.geom_loss_weight * geom_loss
            + self.logit_loss_weight * logit_loss
        )
        return kd_loss, span_loss, geom_loss, logit_loss

    @staticmethod
    def _has_pooler(student_inputs: Dict[str, Any], teacher_inputs: Dict[str, Any]) -> bool:
        return all(
            key in student_inputs and key in teacher_inputs
            for key in ("pooler_safe_idx", "pooler_mask")
        )

    def _span_pooled_distillation_loss(self, distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs):
        span_loss, student_last, teacher_last, span_weights = self._span_pooled_hidden_loss(
            distiller,
            student_outputs,
            teacher_outputs,
            student_inputs,
            teacher_inputs,
        )
        span_weights = span_weights.to(student_last.device)
        pair_weights = _pair_weights(span_weights)
        geom_projector = self._get_projector(distiller, idx=-1)
        geom_loss = _geometry_loss(
            student_last,
            teacher_last.to(student_last.device),
            pair_weights,
            projector=geom_projector,
        )

        student_logits = _get_output_head(distiller.student)(student_last)
        teacher_logits = _get_output_head(distiller.teacher)(teacher_last)
        logit_loss = self._aligned_soft_label_loss(distiller, student_logits, teacher_logits, span_weights.gt(0))

        kd_loss = (
            self.span_loss_weight * span_loss
            + self.geom_loss_weight * geom_loss
            + self.logit_loss_weight * logit_loss
        )
        return kd_loss, span_loss, geom_loss, logit_loss

    def _mapped_layers(self, student_hidden_states, teacher_hidden_states) -> Tuple[list[int], list[int]]:
        student_layers = _as_layer_list(getattr(self.args, "student_layer_mapping", None))
        teacher_layers = _as_layer_list(getattr(self.args, "teacher_layer_mapping", None))
        if not student_layers or not teacher_layers:
            student_layers = [len(student_hidden_states) - 1]
            teacher_layers = [len(teacher_hidden_states) - 1]

        if len(student_layers) != len(teacher_layers):
            raise ValueError("SRE requires student_layer_mapping and teacher_layer_mapping to have the same length.")
        return student_layers, teacher_layers

    def _span_pooled_hidden_loss(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        student_hidden_states = getattr(student_outputs, "hidden_states", None)
        teacher_hidden_states = getattr(teacher_outputs, "hidden_states", None)
        if student_hidden_states is None or teacher_hidden_states is None:
            zero = _last_hidden(student_outputs).new_zeros(())
            student_last = _pool_hidden_states(
                _last_hidden(student_outputs),
                student_inputs["pooler_safe_idx"],
                student_inputs["pooler_mask"],
            )
            teacher_last = _pool_hidden_states(
                _last_hidden(teacher_outputs),
                teacher_inputs["pooler_safe_idx"],
                teacher_inputs["pooler_mask"],
            )
            span_weights = _span_weights(student_inputs["pooler_mask"], self.p).to(student_last.device)
            return zero, student_last, teacher_last, span_weights

        student_layers, teacher_layers = self._mapped_layers(student_hidden_states, teacher_hidden_states)

        losses = []
        student_last = teacher_last = span_weights = None
        for idx, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
            student_hidden, _student_span_weights = _pool_hidden_states_with_weights(
                student_hidden_states[student_layer],
                student_inputs["pooler_safe_idx"],
                student_inputs["pooler_mask"],
                student_inputs.get("attention_mask"),
                _layer_attention(student_outputs, student_layer),
            )
            teacher_hidden, teacher_span_weights = _pool_hidden_states_with_weights(
                teacher_hidden_states[teacher_layer],
                teacher_inputs["pooler_safe_idx"],
                teacher_inputs["pooler_mask"],
                teacher_inputs.get("attention_mask"),
                _layer_attention(teacher_outputs, teacher_layer),
            )
            layer_weights = _normalize_weights(teacher_span_weights, self.p).to(student_hidden.device)
            projector = self._get_projector(distiller, idx)
            losses.append(
                _weighted_cosine_loss(
                    student_hidden,
                    teacher_hidden.to(student_hidden.device),
                    layer_weights,
                    projector=projector,
                )
            )
            if idx == len(student_layers) - 1:
                # Keep student_last in *student* hidden dim so the downstream
                # logit head (student.lm_head) and a separately-applied geometry
                # projector both see compatible inputs.
                student_last = student_hidden
                teacher_last = teacher_hidden
                span_weights = layer_weights

        layer_weights = self._hidden_layer_weights(len(losses), losses[0].device)
        span_loss = sum(weight * loss for weight, loss in zip(layer_weights, losses))
        return span_loss, student_last, teacher_last, span_weights

    def _span_loss(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        token_weights,
        seq_len: Optional[int],
    ) -> torch.Tensor:
        student_hidden_states = getattr(student_outputs, "hidden_states", None)
        teacher_hidden_states = getattr(teacher_outputs, "hidden_states", None)
        if student_hidden_states is None or teacher_hidden_states is None:
            return _last_hidden(student_outputs).new_zeros(())

        student_layers, teacher_layers = self._mapped_layers(student_hidden_states, teacher_hidden_states)

        losses = []
        for idx, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
            student_hidden = student_hidden_states[student_layer][:, -seq_len:]
            teacher_hidden = teacher_hidden_states[teacher_layer][:, -seq_len:]
            projector = self._get_projector(distiller, idx)
            losses.append(
                _weighted_cosine_loss(
                    student_hidden,
                    teacher_hidden.to(student_hidden.device),
                    token_weights,
                    projector=projector,
                )
            )

        layer_weights = self._hidden_layer_weights(len(losses), losses[0].device)
        return sum(weight * loss for weight, loss in zip(layer_weights, losses))
