"""Collator turning precomputed loss masks into label tensors for masked SFT."""

from dataclasses import dataclass

import torch

from masked_loss import IGNORE_INDEX


@dataclass
class MaskedSFTCollator:
    """Pads a batch and derives labels from each example's loss mask.

    Each example provides `input_ids` and `loss_mask` (same length, 1 = supervise).
    Labels are input_ids with IGNORE_INDEX wherever the mask is 0, so the mask is the
    only thing distinguishing the vanilla and spectral runs.
    """

    pad_token_id: int

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)

        input_ids, attention_mask, labels, loss_weights = [], [], [], []
        has_loss_weights = any("loss_weights" in example for example in examples)
        if has_loss_weights and not all("loss_weights" in example for example in examples):
            raise ValueError("a batch must either provide loss_weights for every example or for none")
        for example in examples:
            ids = list(example["input_ids"])
            mask = list(example["loss_mask"])
            if len(ids) != len(mask):
                raise ValueError(
                    f"loss_mask length {len(mask)} != input_ids length {len(ids)}"
                )

            padding = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(
                [token if keep else IGNORE_INDEX for token, keep in zip(ids, mask)]
                + [IGNORE_INDEX] * padding
            )
            if has_loss_weights:
                weights = list(example["loss_weights"])
                if len(weights) != len(ids):
                    raise ValueError(
                        f"loss_weights length {len(weights)} != input_ids length {len(ids)}"
                    )
                if any(weight < 0 for weight in weights):
                    raise ValueError("loss_weights must be non-negative")
                loss_weights.append(weights + [0.0] * padding)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if has_loss_weights:
            batch["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return batch
