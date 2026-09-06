import pytest
from accelerate.data_loader import BatchSamplerShard, prepare_data_loader
from torch.utils.data import DataLoader

from distillation.sampling import (
    TemplateDistributedBatchSampler,
    selector_droppable_batch_range,
)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(2, (10, 5)), (4, (5, 3)), (8, (3, 1))],
)
def test_selector_droppable_batch_range_scales_with_batch_size(
    batch_size: int, expected: tuple[int, int]
) -> None:
    assert selector_droppable_batch_range(
        batch_size=batch_size,
        contrast_pairs=5,
        unpaired_negatives=5,
        total_rows=32 if batch_size > 2 else 30,
    ) == expected


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(2, (4096, 2731)), (4, (2048, 1365)), (8, (1024, 682)), (16, (512, 341))],
)
def test_current_equal_task_layout_excludes_partial_tail_from_droppable_batches(
    batch_size: int, expected: tuple[int, int]
) -> None:
    assert selector_droppable_batch_range(
        batch_size=batch_size,
        contrast_pairs=2048,
        unpaired_negatives=2731,
        total_rows=13654,
    ) == expected


def test_distributed_batches_preserve_contiguous_groups_without_duplicates() -> None:
    dataset = range(11)
    world_size = 2
    batch_size = 4
    sampler = TemplateDistributedBatchSampler(dataset, batch_size=batch_size, num_replicas=world_size)

    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        for rank in range(world_size):
            shard = BatchSamplerShard(
                sampler,
                num_processes=world_size,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
            actual = [list(batch) for batch in shard]
            assert [len(batch) for batch in actual] == [4]
            assert all(batch == list(range(batch[0], batch[0] + len(batch))) for batch in actual)
        batches = [list(batch) for batch in sampler]
        flattened = [index for batch in batches for index in batch]
        assert sorted(flattened) == list(range(8))
        assert len(flattened) == len(set(flattened))


def test_single_process_keeps_incomplete_final_batch_and_prepared_order() -> None:
    dataset = range(10)
    sampler = TemplateDistributedBatchSampler(dataset, batch_size=4, num_replicas=1)
    actual = [list(batch) for batch in sampler]
    assert [len(batch) for batch in actual] == [4, 4, 2]
    assert all(batch == list(range(batch[0], batch[0] + len(batch))) for batch in actual)
    assert sorted(index for batch in actual for index in batch) == list(dataset)


def test_distributed_sampler_preserves_generator_selector_pairs() -> None:
    sampler = TemplateDistributedBatchSampler(range(12), batch_size=2, num_replicas=2)

    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        for rank in range(2):
            shard = BatchSamplerShard(
                sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
            for batch in shard:
                assert len(batch) == 2
                assert batch[0] % 2 == 0
                assert batch[1] == batch[0] + 1


def test_sampler_shuffles_complete_two_batch_contrast_blocks() -> None:
    sampler = TemplateDistributedBatchSampler(range(32), batch_size=2, num_replicas=2, seed=17)

    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        batches = [list(batch) for batch in sampler]
        for offset in range(0, len(batches), 2):
            first, second = batches[offset : offset + 2]
            assert second[0] == first[0] + 2
            assert second[1] == first[1] + 2


def test_accelerate_propagates_epoch_to_wrapped_batch_sampler() -> None:
    epoch_batches = []
    for rank in range(2):
        sampler = TemplateDistributedBatchSampler(range(32), batch_size=2, num_replicas=2, seed=17)
        dataloader = DataLoader(range(32), batch_sampler=sampler)
        prepared = prepare_data_loader(
            dataloader,
            num_processes=2,
            process_index=rank,
            split_batches=False,
            even_batches=False,
        )

        epoch_zero = [batch.tolist() for batch in prepared]
        prepared.set_epoch(1)
        epoch_one = [batch.tolist() for batch in prepared]

        assert sampler.epoch == 1
        assert epoch_one != epoch_zero
        assert all(batch[1] == batch[0] + 1 for batch in epoch_one)
        epoch_batches.append(epoch_one)

    for rank_zero_batch, rank_one_batch in zip(*epoch_batches, strict=True):
        assert rank_one_batch[0] == rank_zero_batch[0] + 2
        assert rank_one_batch[1] == rank_zero_batch[1] + 2


def test_odd_batch_count_rotates_only_droppable_tail_batch() -> None:
    sampler = TemplateDistributedBatchSampler(
        range(30),
        batch_size=2,
        num_replicas=2,
        seed=17,
        droppable_batch_start=4,
        droppable_batch_count=11,
    )
    omitted_by_epoch = []
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        included = {index for batch in sampler for index in batch}
        omitted = set(range(30)).difference(included)
        assert len(omitted) == 2
        assert min(omitted) >= 8
        omitted_by_epoch.append(omitted)

    assert omitted_by_epoch[0] != omitted_by_epoch[1]


@pytest.mark.parametrize(
    ("batch_size", "dataset_size", "droppable_start", "droppable_count"),
    [(4, 36, 5, 3), (8, 56, 3, 1)],
)
def test_larger_batches_drop_only_the_computed_negative_region(
    batch_size: int,
    dataset_size: int,
    droppable_start: int,
    droppable_count: int,
) -> None:
    sampler = TemplateDistributedBatchSampler(
        range(dataset_size),
        batch_size=batch_size,
        num_replicas=2,
        seed=17,
        droppable_batch_start=droppable_start,
        droppable_batch_count=droppable_count,
    )

    included = {index for batch in sampler for index in batch}
    omitted = set(range(dataset_size)).difference(included)

    assert len(omitted) == batch_size
    assert min(omitted) >= droppable_start * batch_size
    assert max(omitted) < (droppable_start + droppable_count) * batch_size
