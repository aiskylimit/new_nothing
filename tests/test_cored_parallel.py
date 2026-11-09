import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cored"))
from core.efficient_loss import step_token_weights, weighted_nll_kl
from core.losses import step_nll, step_token_counts, weighted_step_loss
from train.parallel import backward_with_retry, global_windows, pack_microbatches
from train.pipeline_b200 import merge_prefix_shards, validate_npz_coverage


@pytest.mark.parametrize("chunk", [1, 3, 100])
@pytest.mark.parametrize("with_teacher", [False, True])
def test_exact_loss_and_gradient_match_full_autograd(chunk, with_teacher):
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(2, 7, 11, generator=generator)
    student = source.clone().requires_grad_()
    reference = source.clone().requires_grad_()
    teacher = torch.randn(source.shape, generator=generator) if with_teacher else None
    labels = torch.randint(0, 11, (2, 7), generator=generator)
    weights = torch.rand(2, 7, generator=generator)
    weights[:, :2] = 0  # prompt
    weights[1, -2:] = 0  # padding
    nll, kl = weighted_nll_kl(student, labels, weights, teacher=teacher, chunk=chunk)
    expected_nll = (F.cross_entropy(reference.flatten(0, 1), labels.flatten(), reduction="none")
                    .reshape_as(weights) * weights).sum()
    expected_kl = reference.new_zeros(())
    if teacher is not None:
        log_t, log_s = teacher.log_softmax(-1), reference.log_softmax(-1)
        expected_kl = ((log_t.exp() * (log_t - log_s)).sum(-1) * weights).sum()
    torch.testing.assert_close(nll, expected_nll)
    torch.testing.assert_close(kl, expected_kl)
    (nll + 0.7 * kl).backward()
    (expected_nll + 0.7 * expected_kl).backward()
    torch.testing.assert_close(student.grad, reference.grad, atol=3e-7, rtol=3e-6)
    assert torch.all(student.grad[weights == 0] == 0)


def test_step_weights_match_legacy_reduction_including_empty_segments():
    torch.manual_seed(12)
    logits = torch.randn(9, 17, requires_grad=True)
    labels = torch.randint(0, 17, (9,))
    spans = [(0, 3), (3, 3), (4, 7), (7, 9)]
    weights = torch.tensor([0.2, 0.7, 1.3, 1.0])
    coef = step_token_weights(9, spans, weights, device="cpu")
    fast, _ = weighted_nll_kl(logits, torch.roll(labels, -1), coef, chunk=2)
    old = weighted_step_loss(step_nll(logits, labels, spans), step_token_counts(spans), weights, "step_mean")
    torch.testing.assert_close(fast, old)


def test_zero_supervised_tokens_returns_zero_loss_and_gradient():
    logits = torch.randn(1, 3, 7, requires_grad=True)
    nll, kl = weighted_nll_kl(logits, torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3))
    (nll + kl).backward()
    assert nll.item() == kl.item() == 0
    assert torch.all(logits.grad == 0)


def test_oom_retry_rolls_back_partial_gradients_and_rng():
    param = torch.nn.Parameter(torch.tensor(2.0))
    param.grad = torch.tensor(7.0)
    torch.manual_seed(5)
    expected_random = torch.rand(4).sum().item()
    torch.manual_seed(5)

    def partial(items):
        value = torch.rand(())
        (param * sum(items)).backward()
        if len(items) > 1:
            raise torch.OutOfMemoryError("simulated OOM after first expert backward")
        return {"random": value.item(), "examples": len(items)}

    result, retries = backward_with_retry([1, 2, 3, 4], partial, [param], torch.device("cpu"))
    assert param.grad.item() == 17
    assert result["examples"] == 4
    assert result["random"] == pytest.approx(expected_random)
    assert retries == 3


def test_global_batches_do_not_pad_repeat_or_drop_partial_shards():
    order = list(range(67))
    ranks = [list(global_windows(order, 33, rank, 3)) for rank in range(3)]
    all_ids = []
    for window in range(3):
        grouped = [ranks[rank][window] for rank in range(3)]
        expected_count = 33 if window < 2 else 1
        assert all(count == expected_count for _, count in grouped)
        merged = [item for ids, _ in grouped for item in ids]
        assert len(merged) == expected_count
        all_ids.extend(merged)
    assert sorted(all_ids) == order
    assert ranks[1][-1][0] == ranks[2][-1][0] == []


def test_token_budget_bucketing_covers_every_example():
    examples = [SimpleNamespace(input_ids=torch.ones(length)) for length in [2, 7, 3, 8, 1]]
    batches = list(pack_microbatches(examples, 3, 15))
    assert sum(map(len, batches)) == len(examples)
    assert all(len(batch) * max(x.input_ids.numel() for x in batch) <= 15 for batch in batches)


def test_length_balancing_keeps_global_batch_and_limits_rank_counts():
    lengths = [100, 99, 98, 3, 2, 1]
    shards = [next(global_windows(list(range(6)), 6, r, 3, lengths))[0] for r in range(3)]
    assert sorted(i for shard in shards for i in shard) == list(range(6))
    assert all(len(shard) == 2 for shard in shards)
    assert all(sum(lengths[i] > 90 for i in shard) == 1 for shard in shards)


def test_prefix_merge_preserves_order_and_rejects_duplicates(tmp_path):
    rows = [{"id": f"s{i}", "thinking": str(i)} for i in range(7)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    parts = []
    for rank in range(3):
        path = tmp_path / f"part-{rank}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows[rank::3]))
        parts.append(path)
    output = tmp_path / "merged.jsonl"
    merge_prefix_shards(source, parts, output)
    assert output.read_text() == source.read_text()
    with parts[0].open("a") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        merge_prefix_shards(source, parts, output)


def test_npz_cache_requires_exact_coverage_and_finite_values(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id": "s0"}\n')
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="missing=1"):
        validate_npz_coverage(source, cache, ["w"])
    np.savez(cache / "s0.npz", w=np.ones(2))
    validate_npz_coverage(source, cache, ["w"])
    np.savez(cache / "s0.npz", w=np.array([float("nan")]))
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_npz_coverage(source, cache, ["w"])
