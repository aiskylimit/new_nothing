"""Shared masked-SFT infrastructure: dataset + Modal volume-commit callback.

Framework-agnostic pieces used by both the DeepSpeed HF-Trainer path (`train_sft.py`) and the
Unsloth single-GPU full-FT path (`train_sft_unsloth.py`). Kept here so the Unsloth process does
not have to import `train_sft` (and its peft/DeepSpeed stack, which can clash with Unsloth's
patched transformers).
"""

import json

from torch.utils.data import Dataset
from transformers import TrainerCallback


class MaskedSFTDataset(Dataset):
    """JSONL records of {input_ids, loss_mask} produced by build_masks.py."""

    def __init__(self, path: str, max_seq_len: int | None = None):
        with open(path) as handle:
            self.records = [json.loads(line) for line in handle]
        if max_seq_len is not None:
            before = len(self.records)
            self.records = [r for r in self.records if len(r["input_ids"]) <= max_seq_len]
            dropped = before - len(self.records)
            if dropped:
                print(f"--max-seq-len {max_seq_len}: dropped {dropped}/{before} samples (too long to fit in GPU memory at batch size 1)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def supervised_token_count(self) -> int:
        return sum(sum(record["loss_mask"]) for record in self.records)


class VolumeCommitCallback(TrainerCallback):
    """Persist a mounted Modal Volume after every checkpoint save.

    Modal only makes a Volume's writes durable/visible when the function exits (or on an explicit
    commit), so a crash mid-run would otherwise lose every finished epoch. Committing on each save
    means a later re-run can resume from the last completed epoch instead of starting over. Only
    activated on Modal (env MODAL_COMMIT_VOLUME=<volume name>); a no-op everywhere else, so the
    B200/offline path is untouched.
    """

    def __init__(self, volume_name: str):
        # Degrade gracefully: if modal isn't importable (e.g. this env doesn't ship it), disable the
        # callback rather than crash training -- durability is a nice-to-have, finishing the run isn't.
        self._vol = None
        try:
            import modal

            self._vol = modal.Volume.from_name(volume_name)
        except Exception as exc:
            print(f"[volume-commit] disabled ({exc}); training continues without per-epoch commits", flush=True)

    def on_save(self, args, state, control, **kwargs):
        if self._vol is None:
            return
        try:
            self._vol.commit()
            print(
                f"[volume-commit] committed after step {state.global_step} "
                f"(epoch {state.epoch:.2f})",
                flush=True,
            )
        except Exception as exc:  # a failed commit must never abort training
            print(f"[volume-commit] WARN commit failed: {exc}", flush=True)
