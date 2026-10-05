"""Collator turning precomputed loss masks into label tensors for masked SFT."""

from dataclasses import dataclass

import torch

from sgl.training.losses import IGNORE_INDEX


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
        if any("token_source" in example for example in examples):
            # provenance of each token (sgl.transforms.provenance): 0 = unsupervised/prompt,
            # 1 = teacher prefix, 2 = continuation; padding is 0.
            if not all("token_source" in example for example in examples):
                raise ValueError("a batch must either provide token_source for every example or for none")
            rows = []
            for example in examples:
                source = list(example["token_source"])
                if len(source) != len(example["input_ids"]):
                    raise ValueError(
                        f"token_source length {len(source)} != input_ids length {len(example['input_ids'])}"
                    )
                rows.append(source + [0] * (max_length - len(source)))
            batch["token_source"] = torch.tensor(rows, dtype=torch.long)
        if any("step_id" in example for example in examples):
            batch.update(self._transition_fields(examples, max_length))
        return batch

    @staticmethod
    def _transition_fields(examples: list[dict], max_length: int) -> dict[str, torch.Tensor]:
        """L_trans structure from sgl.transforms.build_transitions, padded with -1 (step_id: outside the
        CoT; step_end / pair_src: no entry) so the trainer can filter per example."""
        if not all("step_id" in example for example in examples):
            raise ValueError("a batch must either provide step_id for every example or for none")

        def pad(rows: list[list[int]]) -> torch.Tensor:
            width = max(1, max(len(row) for row in rows))
            return torch.tensor([row + [-1] * (width - len(row)) for row in rows], dtype=torch.long)

        for example in examples:
            if len(example["step_id"]) != len(example["input_ids"]):
                raise ValueError(
                    f"step_id length {len(example['step_id'])} != input_ids length {len(example['input_ids'])}"
                )
        return {
            "step_id": pad([list(example["step_id"]) for example in examples]),
            "step_end": pad([list(example["step_end"]) for example in examples]),
            "pair_src": pad([list(example["pair_src"]) for example in examples]),
            "num_steps": torch.tensor([int(example["num_steps"]) for example in examples], dtype=torch.long),
        }
