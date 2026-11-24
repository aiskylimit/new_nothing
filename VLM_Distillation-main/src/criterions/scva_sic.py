"""BeD-VLM: supervised generation with SCVA and semantic interventions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.criterions.scva import IGNORE_INDEX, SCVACriterion, _patch_centers


class DecoderReplay:
    """Reuse the decoder inputs from the full multimodal forward."""

    def __init__(self, model):
        encoder = model.encoder
        if hasattr(encoder, "get_base_model"):
            encoder = encoder.get_base_model()
        self.decoder = getattr(getattr(encoder, "model", None), "language_model", None)
        if self.decoder is None:
            raise ValueError("SIC requires a backbone with a separate multimodal text decoder.")
        self.lm_head = encoder.get_output_embeddings()
        self.token_embeddings = encoder.get_input_embeddings().weight

    def full_forward(self, model, inputs):
        def capture(_module, _args, kwargs):
            self.decoder_inputs = kwargs.copy()
            if model.training:
                # Reuse the full pass's dropout masks for each intervention.
                self.cpu_rng = torch.random.get_rng_state()
                device = kwargs["inputs_embeds"].device
                self.cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            else:
                self.cpu_rng = None
                self.cuda_rng = None

        handle = self.decoder.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            return model(**inputs)
        finally:
            handle.remove()

    def masked_logits(self, embeddings, mask, batch_idx, positions):
        kwargs = dict(self.decoder_inputs, inputs_embeds=embeddings.masked_fill(mask[..., None], 0),
                      output_attentions=False, output_hidden_states=False, return_dict=True)
        # Qwen3 experimental recipes also carry visual features into early decoder layers.
        if kwargs.get("deepstack_visual_embeds") is not None:
            deepstack_mask = mask[kwargs["visual_pos_masks"]]
            kwargs["deepstack_visual_embeds"] = [
                features.masked_fill(deepstack_mask[:, None], 0)
                for features in kwargs["deepstack_visual_embeds"]
            ]
        devices = [embeddings.device] if self.cuda_rng is not None else []
        with torch.random.fork_rng(devices=devices, enabled=self.cpu_rng is not None):
            if self.cpu_rng is not None:
                torch.random.set_rng_state(self.cpu_rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state(self.cuda_rng, embeddings.device)
            hidden = self.decoder(**kwargs).last_hidden_state
        return self.lm_head(hidden[batch_idx, positions])


@torch.no_grad()
def intervention_groups(region, teacher_hidden, budget, spatial_weight):
    """Merge nearest semantic/spatial centroids until the intervention budget fits."""
    active = region.student_assignment.any(dim=0).nonzero(as_tuple=True)[0].tolist()
    groups = [[index] for index in active]
    if len(groups) <= budget:
        return groups

    assignment = region.teacher_assignment[:, active].T
    coords = _patch_centers(assignment.size(1), *region.teacher_grid, teacher_hidden.device)
    semantic_sums = assignment @ F.normalize(teacher_hidden.float(), dim=-1)
    spatial_sums = assignment @ coords
    counts = assignment.sum(dim=-1)

    while len(groups) > budget:
        semantic = F.normalize(semantic_sums, dim=-1)
        spatial = spatial_sums / counts[:, None]
        distances = 1 - semantic @ semantic.T + spatial_weight * torch.cdist(spatial, spatial)
        distances.masked_fill_(torch.ones_like(distances, dtype=torch.bool).triu(), torch.inf)
        right, left = divmod(int(distances.argmin()), len(groups))
        groups[left].extend(groups.pop(right))
        semantic_sums[left] += semantic_sums[right]
        spatial_sums[left] += spatial_sums[right]
        counts[left] += counts[right]
        keep = torch.arange(len(groups) + 1, device=counts.device).ne(right)
        semantic_sums, spatial_sums, counts = semantic_sums[keep], spatial_sums[keep], counts[keep]
    return groups


def response_positions(labels):
    """Select the causal logit position that predicts each supervised token."""
    return labels[1:].ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]


def embedded_shift(full_log_probs, masked_logits, token_embeddings):
    masked_log_probs = F.log_softmax(masked_logits.float(), dim=-1)
    shift = (full_log_probs - masked_log_probs).to(token_embeddings.dtype)
    return (shift @ token_embeddings).float()


class SCVASICCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scva = SCVACriterion(args)
        self.ce_weight = args.scva_sic_ce_weight
        self.lambda_v = args.scva_sic_lambda_v
        self.lambda_s = args.scva_sic_lambda_sic
        self.smoothing = args.scva_region_smoothing
        self.budget = args.sic_max_clusters or args.sic_num_regions
        self.use_projector = args.sic_use_projector
        if self.budget < 1 or self.smoothing <= 0:
            raise ValueError("SIC requires a positive region budget and SCVA requires positive smoothing.")
        if min(self.ce_weight, self.lambda_v, self.lambda_s) < 0:
            raise ValueError("BeD-VLM loss weights must be nonnegative.")

    def forward(self, distiller, batch):
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch["teacher_inputs"]
        student_replay = DecoderReplay(distiller.student) if self.lambda_s else None
        teacher_replay = DecoderReplay(distiller.teacher) if self.lambda_s else None
        student = (student_replay.full_forward(distiller.student, student_inputs)
                   if student_replay else distiller.student(**student_inputs))
        with torch.no_grad():
            teacher = (teacher_replay.full_forward(distiller.teacher, teacher_inputs)
                       if teacher_replay else distiller.teacher(**teacher_inputs))
        ce = student.loss
        if ce is None:
            raise RuntimeError("BeD-VLM requires assistant response labels.")

        regions = self.scva.build_regions(student, teacher, student_inputs, teacher_inputs)
        scva = ce.new_zeros(())
        if self.lambda_v:
            scva = self.scva._scva_loss(
                student, teacher, student_inputs, teacher_inputs, regions=regions,
                layer_pairs=distiller.scva_layer_pairs, smoothing=self.smoothing,
            )
        sic = ce.new_zeros(())
        active = 0
        if self.lambda_s:
            projector = distiller.projectors["sic"] if self.use_projector else None
            sic, active = self._sic_loss(
                student, teacher, student_inputs["labels"], teacher_inputs["labels"], regions,
                student_replay, teacher_replay, projector,
            )

        metrics = {
            "loss": self.ce_weight * ce + self.lambda_v * scva + self.lambda_s * sic,
            "hard_loss": ce.detach(),
            "scva_loss": scva.detach(),
            "sic_loss": sic.detach(),
            "sic_active_regions": ce.new_tensor(active),
        }
        if self.lambda_v:
            metrics.update(self.scva._scalarize_counts(ce))
        return metrics

    def _sic_loss(self, student, teacher, s_labels, t_labels, regions, s_replay, t_replay, projector):
        sample_losses = []
        active = 0
        for b, region in regions.items():
            s_pos = response_positions(s_labels[b])
            t_pos = response_positions(t_labels[b])
            length = min(s_pos.numel(), t_pos.numel())
            if not length:
                continue
            s_pos, t_pos = s_pos[:length], t_pos[:length]
            groups = intervention_groups(
                region, teacher.hidden_states[-1][b, region.teacher_positions],
                self.budget, self.scva.spatial_weight,
            )
            if not groups:
                continue
            s_full = F.log_softmax(student.logits[b, s_pos].float(), dim=-1)
            with torch.no_grad():
                t_full = F.log_softmax(teacher.logits[b, t_pos].float(), dim=-1)
                entropy = -(t_full.exp() * t_full).sum(dim=-1)
                weights = (-entropy / math.log(t_full.size(-1))).exp()
            s_embeds = s_replay.decoder_inputs["inputs_embeds"]
            t_embeds = t_replay.decoder_inputs["inputs_embeds"]
            region_losses = []
            for group in groups:
                t_mask = self._mask(t_embeds, b, region.teacher_positions, region.teacher_assignment, group)
                s_mask = self._mask(s_embeds, b, region.student_positions, region.student_assignment, group)
                with torch.no_grad():
                    t_logits = t_replay.masked_logits(t_embeds, t_mask, b, t_pos)
                    t_shift = embedded_shift(t_full, t_logits, t_replay.token_embeddings)

                def loss(embeddings, full, mask, target, batch_idx=b, positions=s_pos):
                    logits = s_replay.masked_logits(embeddings, mask, batch_idx, positions)
                    shift = embedded_shift(full, logits, s_replay.token_embeddings)
                    if projector is not None:
                        shift = projector(shift.to(projector.weight.dtype)).float()
                    distance = 1 - F.cosine_similarity(shift, target, dim=-1, eps=1e-6)
                    return distance * target.norm(dim=-1).gt(1e-6)

                distance = checkpoint(loss, s_embeds, s_full, s_mask, t_shift,
                                      use_reentrant=False, preserve_rng_state=False)
                region_losses.append((weights * distance).sum())
            sample_losses.append(torch.stack(region_losses).sum() / weights.sum())
            active += len(groups)

        if sample_losses:
            return torch.stack(sample_losses).mean(), active
        zero = student.logits.new_zeros(())
        if projector is not None:
            zero = zero + 0 * projector.weight.flatten()[0]
        return zero, active

    @staticmethod
    def _mask(embeddings, batch_idx, positions, assignment, group):
        mask = torch.zeros(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        mask[batch_idx, positions] = assignment[:, group].any(dim=-1)
        return mask
