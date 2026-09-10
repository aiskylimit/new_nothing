from __future__ import annotations

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import BatchSampler, Sampler


class _EpochAwareSequentialSampler(Sampler[int]):
    """Expose epoch state where Accelerate expects to propagate it."""

    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _distributed_batch_count(size: int, batch_size: int, num_replicas: int) -> int:
    full_batches, remainder = divmod(size, batch_size)
    if num_replicas == 1:
        return full_batches + int(remainder > 0)
    return (full_batches // num_replicas) * num_replicas


def selector_droppable_batch_range(
    *,
    batch_size: int,
    contrast_pairs: int,
    unpaired_negatives: int,
    total_rows: int,
) -> tuple[int, int]:
    """Locate full prepared batches containing only unpaired selector rows.

    Preparation assigns half of each mixed batch to selector rows. Contrast
    rows precede unpaired negatives, so a boundary batch containing both is
    protected and only the subsequent negative-only batches are droppable.
    """

    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size must be a positive even number.")
    if contrast_pairs < 0 or unpaired_negatives < 0 or total_rows < 0:
        raise ValueError("Selector layout counts must be non-negative.")
    selector_capacity = batch_size // 2
    contrast_rows = contrast_pairs * 2
    selector_rows = contrast_rows + unpaired_negatives
    if total_rows < selector_rows:
        raise ValueError("Prepared row count cannot be smaller than its selector row count.")
    first_negative_only_batch = (contrast_rows + selector_capacity - 1) // selector_capacity
    selector_batch_count = (selector_rows + selector_capacity - 1) // selector_capacity
    # The final mixed chunk can be a partial dataloader batch when generator
    # and selector row counts are equal but selector_capacity does not divide
    # selector_rows. Distributed sampling never yields that remainder, so it
    # must not be advertised as a full droppable batch.
    full_batch_count = total_rows // batch_size
    droppable_start = min(first_negative_only_batch, full_batch_count)
    droppable_stop = min(selector_batch_count, full_batch_count)
    return droppable_start, max(0, droppable_stop - droppable_start)


def _preserved_distributed_batches(
    indices: list[int],
    *,
    batch_size: int,
    num_replicas: int,
    seed: int,
    epoch: int,
    droppable_batch_start: int = 0,
    droppable_batch_count: int = 0,
) -> Iterator[list[int]]:
    """Shuffle prepared contrast blocks without shuffling their samples.

    The prepared dataset already encodes the intended generator/selector batch
    composition and same-question selector pairs. A block contains two batches
    only when the per-batch selector capacity is odd, which prevents an adjacent
    YES/NO pair from being separated by shuffling. Otherwise each prepared batch
    is independently movable. At batch size two with two replicas, this also
    places both sides of a contrast pair in the same distributed micro-step.
    """

    full_batch_count, remainder = divmod(len(indices), batch_size)
    batches = [indices[start * batch_size : (start + 1) * batch_size] for start in range(full_batch_count)]
    drop_count = full_batch_count % num_replicas if num_replicas > 1 else 0
    if drop_count:
        droppable_stop = droppable_batch_start + droppable_batch_count
        if (
            droppable_batch_start < 0
            or droppable_stop > full_batch_count
            or droppable_batch_count < drop_count
        ):
            raise ValueError(
                "Distributed sampling needs enough explicitly droppable non-contrast batches "
                "to equalize rank lengths."
            )
        first_offset = (epoch * drop_count) % droppable_batch_count
        dropped = {
            droppable_batch_start + (first_offset + offset) % droppable_batch_count
            for offset in range(drop_count)
        }
        batches = [batch for index, batch in enumerate(batches) if index not in dropped]
        full_batch_count = len(batches)
    generator = torch.Generator().manual_seed(seed + epoch)
    selector_capacity = batch_size // 2
    shuffle_block_size = 2 if selector_capacity % 2 else 1
    complete_block_count, trailing_batch_count = divmod(full_batch_count, shuffle_block_size)
    blocks = [
        batches[start * shuffle_block_size : (start + 1) * shuffle_block_size]
        for start in range(complete_block_count)
    ]
    block_order = torch.randperm(complete_block_count, generator=generator).tolist()
    ordered_batches = [batch for block_index in block_order for batch in blocks[block_index]]
    if trailing_batch_count:
        # Keep an incomplete contrast block last so it cannot shift the
        # distributed-rank alignment of any preceding contrast pair.
        ordered_batches.extend(batches[-trailing_batch_count:])
    if num_replicas > 1 and full_batch_count % num_replicas:
        raise AssertionError("Distributed batch count was not equalized across ranks")
    yield from ordered_batches
    if num_replicas == 1 and remainder:
        # Accelerate accepts a single-process incomplete batch only at the end.
        yield indices[full_batch_count * batch_size :]


class TemplateDistributedBatchSampler(BatchSampler):
    """Distribute prepared contrast blocks without destroying their composition.

    With ``even_batches=False``, ranks have equal step counts and no examples
    are duplicated. Only complete contrast-safe blocks and their rank ownership
    are shuffled.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        num_replicas: int,
        seed: int = 0,
        droppable_batch_start: int = 0,
        droppable_batch_count: int = 0,
    ) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be a positive even number.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        self._index_sampler = _EpochAwareSequentialSampler(dataset)
        super().__init__(self._index_sampler, batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.droppable_batch_start = droppable_batch_start
        self.droppable_batch_count = droppable_batch_count

    @property
    def epoch(self) -> int:
        return self._index_sampler.epoch

    def __iter__(self) -> Iterator[list[int]]:
        yield from _preserved_distributed_batches(
            list(range(len(self.dataset))),
            batch_size=self.batch_size,
            num_replicas=self.num_replicas,
            seed=self.seed,
            epoch=self.epoch,
            droppable_batch_start=self.droppable_batch_start,
            droppable_batch_count=self.droppable_batch_count,
        )

    def __len__(self) -> int:
        return _distributed_batch_count(len(self.dataset), self.batch_size, self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self._index_sampler.set_epoch(epoch)
