"""Shared masked-SFT dataset.

Framework-agnostic, used by both the DeepSpeed HF-Trainer path (`sgl.training.train`) and the
Unsloth single-GPU path (`sgl.training.backends.unsloth`). Kept here so the Unsloth process does
not have to import `sgl.training.train` (and its peft/DeepSpeed stack, which can clash with
Unsloth's patched transformers).
"""

import json

from torch.utils.data import Dataset


class MaskedSFTDataset(Dataset):
    """JSONL records of {input_ids, loss_mask} produced by sgl.selection.build_masks."""

    def __init__(self, path: str, max_seq_len: int | None = None):
        with open(path) as handle:
            self.records = [json.loads(line) for line in handle]
        if max_seq_len is not None:
            before = len(self.records)
            self.records = [r for r in self.records if len(r["input_ids"]) <= max_seq_len]
            dropped = before - len(self.records)
            if dropped:
                print(
                    f"--max-seq-len {max_seq_len}: dropped {dropped}/{before} samples "
                    "(too long to fit in GPU memory at batch size 1)"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def supervised_token_count(self) -> int:
        return sum(sum(record["loss_mask"]) for record in self.records)
