from __future__ import annotations

import pytest
import torch

from distillation.task_balancing import (
    GENERATOR_TASK_ID,
    SELECTOR_TASK_ID,
    TaskAwareDataCollator,
    infer_task_id,
    task_balanced_loss,
)


class _Tokenizer:
    _decoded = {
        (1, 99): '{"label": "YES"}',
        (2, 99): '{"label": "NO"}',
        (3, 4, 99): "MATCH (n) RETURN n",
        (5, 99): "YES",
        (6, 99): '{"label": "YES", "reason": "needed"}',
    }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return self._decoded[tuple(token_ids)]


def test_task_inference_uses_exact_selector_protocol() -> None:
    tokenizer = _Tokenizer()
    assert infer_task_id({"labels": [-100, 1, 99]}, tokenizer) == SELECTOR_TASK_ID
    assert infer_task_id({"labels": [-100, 2, 99]}, tokenizer) == SELECTOR_TASK_ID
    assert infer_task_id({"labels": [-100, 3, 4, 99]}, tokenizer) == GENERATOR_TASK_ID
    assert infer_task_id({"labels": [-100, 5, 99]}, tokenizer) == SELECTOR_TASK_ID
    assert infer_task_id({"labels": [-100, 6, 99]}, tokenizer) == GENERATOR_TASK_ID


def test_explicit_rollout_task_id_takes_precedence_over_generated_text() -> None:
    feature = {"labels": torch.tensor([-100, 3, 4, 99]), "task_id": torch.tensor(1)}
    assert infer_task_id(feature, _Tokenizer()) == SELECTOR_TASK_ID


def test_task_aware_collator_removes_feature_metadata_before_delegating() -> None:
    observed = {}

    def delegate(features):
        observed["features"] = features
        return {"labels": torch.tensor([item["labels"] for item in features])}

    collator = TaskAwareDataCollator(delegate, _Tokenizer())
    batch = collator(
        [
            {"labels": [-100, 3, 4, 99]},
            {"labels": [-100, 3, 4, 99], "task_id": torch.tensor(1)},
        ]
    )

    assert all("task_id" not in feature for feature in observed["features"])
    torch.testing.assert_close(batch["task_ids"], torch.tensor([0, 1]))


def test_task_balanced_loss_gives_equal_task_weight_despite_token_scale() -> None:
    task_ids = torch.tensor([GENERATOR_TASK_ID, GENERATOR_TASK_ID, SELECTOR_TASK_ID])
    per_row_loss = torch.tensor([10.0, 20.0, 2.0], requires_grad=True)

    actual, components = task_balanced_loss(
        task_ids,
        selector_weight=0.5,
        loss_fn=lambda mask: per_row_loss[mask].mean(),
    )

    assert components["generator"].item() == pytest.approx(15.0)
    assert components["selector"].item() == pytest.approx(2.0)
    assert actual.item() == pytest.approx(8.5)


def test_single_task_batch_keeps_full_loss_scale() -> None:
    task_ids = torch.tensor([SELECTOR_TASK_ID, SELECTOR_TASK_ID])
    losses = torch.tensor([2.0, 4.0])
    actual, _ = task_balanced_loss(task_ids, 0.5, lambda mask: losses[mask].mean())
    assert actual.item() == pytest.approx(3.0)
