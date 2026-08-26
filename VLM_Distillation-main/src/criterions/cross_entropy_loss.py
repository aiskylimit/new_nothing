from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


class CrossEntropyLoss(nn.Module):
    def __init__(self, args, padding_id: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.args = args
        self.padding_id = padding_id
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)

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
        if self.label_smoothing > 0:
            return F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=self.padding_id,
                reduction="sum",
                label_smoothing=self.label_smoothing,
            )
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.padding_id,
            reduction="sum",
        )

    def compute_token_correct(self, logits: torch.Tensor, target: torch.Tensor, shift: bool = True) -> torch.Tensor:
        if shift:
            logits, target = self.shift_logits_and_labels(logits, target)
        target = target.to(device=logits.device)
        mask = target.ne(self.padding_id)
        return (logits.argmax(dim=-1).eq(target) & mask).sum()

    def compute_token_count(self, logits: torch.Tensor, target: torch.Tensor, shift: bool = True) -> torch.Tensor:
        if shift:
            _logits, target = self.shift_logits_and_labels(logits, target)
        target = target.to(device=logits.device)
        return target.ne(self.padding_id).sum()

    def compute_token_accuracy(self, logits: torch.Tensor, target: torch.Tensor, shift: bool = True) -> torch.Tensor:
        correct = self.compute_token_correct(logits, target, shift=shift).float()
        total = self.compute_token_count(logits, target, shift=shift).float().clamp_min(1.0)
        return correct / total

    def shift_logits_and_labels(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim < 3 or target.ndim < 2:
            raise ValueError("Expected logits with shape (batch, seq, vocab) and labels with shape (batch, seq).")
        seq_len = min(logits.shape[1], target.shape[1])
        if seq_len <= 1:
            return logits[:, :0], target[:, :0]
        logits = logits[:, :seq_len - 1].contiguous()
        target = target[:, 1:seq_len].contiguous()
        return logits, target

    def shift_inputs_for_causal_targets(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = min(input_ids.shape[1], labels.shape[1])
        if seq_len <= 1:
            empty_inputs = input_ids[:, :0]
            empty_labels = labels[:, :0]
            return empty_inputs, empty_labels, empty_labels.ne(self.padding_id)
        shifted_input_ids = input_ids[:, :seq_len - 1].contiguous()
        shifted_labels = labels[:, 1:seq_len].contiguous()
        return shifted_input_ids, shifted_labels, shifted_labels.ne(self.padding_id)

    def teacher_targets(
        self,
        teacher_inputs: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = teacher_inputs.get("labels")
        if labels is not None:
            labels = labels.to(device=device)
            return labels, labels.ne(self.padding_id)

        input_ids = teacher_inputs["input_ids"].to(device=device)
        attention_mask = teacher_inputs.get("attention_mask")
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            mask = attention_mask.to(device=device, dtype=torch.bool)
        labels = input_ids.masked_fill(~mask, self.padding_id)
        return labels, mask
