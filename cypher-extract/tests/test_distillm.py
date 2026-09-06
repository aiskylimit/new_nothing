from types import SimpleNamespace

import pytest
import torch

from distillation.distillm import (
    AdaptiveRolloutScheduler,
    ReplayBuffer,
    RolloutSource,
    StudentRolloutGenerator,
)


def _feature(token: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([token, token + 1]),
        "attention_mask": torch.ones(2, dtype=torch.long),
        "labels": torch.tensor([-100, token + 1]),
    }


def test_replay_buffer_stores_cpu_copies_and_samples_batch() -> None:
    buffer = ReplayBuffer(3, seed=5)
    source = _feature(1)
    buffer.add(source)
    source["input_ids"][0] = 99
    buffer.extend([_feature(3), _feature(5)])
    sample = buffer.sample(2)
    assert len(sample) == 2
    assert all(item["input_ids"].device.type == "cpu" for item in sample)
    assert all(99 not in item["input_ids"].tolist() for item in sample)


def test_replay_state_round_trip_preserves_contents() -> None:
    first = ReplayBuffer(4, seed=3)
    first.extend([_feature(1), _feature(3), _feature(5)])
    state = first.state_dict()
    second = ReplayBuffer(4, seed=999)
    second.load_state_dict(state)
    assert len(second) == 3
    first_sample = first.sample(2)
    second_sample = second.sample(2)
    for left, right in zip(first_sample, second_sample, strict=True):
        torch.testing.assert_close(left["input_ids"], right["input_ids"])


def test_replay_buffer_preserves_task_identity() -> None:
    buffer = ReplayBuffer(2)
    feature = _feature(1)
    feature["task_id"] = torch.tensor(1)
    buffer.add(feature)
    assert buffer.sample(1)[0]["task_id"].item() == 1


def test_scheduler_first_validation_uses_reference_zero_baseline() -> None:
    scheduler = AdaptiveRolloutScheduler(threshold=0.0, loss_eps=0.1, seed=1)
    assert scheduler.previous_validation_loss == pytest.approx(0.0)
    assert scheduler.update(0.16) is True
    assert scheduler.threshold == pytest.approx(0.1)
    assert scheduler.update(0.20) is False
    assert scheduler.threshold == pytest.approx(0.1)


def test_scheduler_improvement_does_not_replace_comparison_baseline() -> None:
    scheduler = AdaptiveRolloutScheduler(threshold=0.0, loss_eps=0.1, seed=1)
    assert scheduler.update(0.5) is True
    assert scheduler.update(0.3) is False
    assert scheduler.update(0.55) is False
    assert scheduler.previous_validation_loss == pytest.approx(0.5)


def test_scheduler_state_round_trip_preserves_threshold_baseline_and_rng() -> None:
    first = AdaptiveRolloutScheduler(threshold=0.0, loss_eps=0.1, seed=3)
    assert first.update(0.5) is True
    first.choose(progress=0.25, replay_size=2, capacity=10, batch_size=2)

    second = AdaptiveRolloutScheduler(threshold=0.0, loss_eps=0.1, seed=999)
    second.load_state_dict(first.state_dict())

    assert second.threshold == pytest.approx(first.threshold)
    assert second.previous_validation_loss == pytest.approx(first.previous_validation_loss)
    for _ in range(5):
        expected = first.choose(progress=0.25, replay_size=2, capacity=10, batch_size=2)
        actual = second.choose(progress=0.25, replay_size=2, capacity=10, batch_size=2)
        assert actual is expected


def test_scheduler_fills_buffer_with_fresh_rollouts_before_replay() -> None:
    scheduler = AdaptiveRolloutScheduler(threshold=1.0, replay_ratio="constant", seed=0)
    assert scheduler.choose(progress=0.5, replay_size=3, capacity=10, batch_size=2) is RolloutSource.FRESH


def test_rollout_source_string_form_is_its_serialized_value() -> None:
    assert str(RolloutSource.FRESH) == "fresh"
    assert RolloutSource.FRESH.value == "fresh"


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _GenerateModel(torch.nn.Module):
    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        response = torch.tensor([[7, 2], [8, 2]], device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat((input_ids, response), dim=1))


class _ImmediateEosModel(torch.nn.Module):
    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        eos = torch.full((input_ids.shape[0], 1), 2, device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat((input_ids, eos), dim=1))


def test_student_rollout_keeps_full_chat_context_and_masks_it() -> None:
    generator = StudentRolloutGenerator(
        tokenizer=_Tokenizer(),
        cutoff_len=8,
        rollout_context_length=5,
        do_sample=False,
    )
    inputs = {
        "input_ids": torch.tensor([[10, 2, 12, 20, 2, 2], [2, 30, 31, 40, 2, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, -100, 20, 2, -100], [-100, -100, -100, 40, 2, -100]]),
        "task_ids": torch.tensor([0, 1]),
    }
    features = generator.generate(_GenerateModel(), inputs)
    assert generator.generation_config.pad_token_id == _Tokenizer.pad_token_id
    assert features[0]["input_ids"].tolist() == [10, 2, 12, 7]
    assert features[0]["labels"].tolist() == [-100, -100, -100, 7]
    assert features[1]["input_ids"].tolist() == [30, 31, 8]
    assert features[1]["labels"].tolist() == [-100, -100, 8]
    assert [feature["task_id"].item() for feature in features] == [0, 1]


def test_student_rollout_extracts_every_assistant_target_span() -> None:
    generator = StudentRolloutGenerator(
        tokenizer=_Tokenizer(),
        cutoff_len=8,
        rollout_context_length=5,
        do_sample=False,
    )
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 20, 21, 30, 31, 40]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        # Function-call tokens 20/21 and a later final-assistant token 40
        # are separated by masked tool-observation context 30/31.
        "labels": torch.tensor([[-100, -100, -100, 20, 21, -100, -100, 40]]),
    }
    prompts = generator.extract_prompts(inputs)
    assert [prompt.tolist() for prompt in prompts] == [[10, 11, 12], [12, 20, 21, 30, 31]]


def test_student_rollout_discards_immediate_eos_without_creating_empty_labels() -> None:
    generator = StudentRolloutGenerator(
        tokenizer=_Tokenizer(),
        cutoff_len=8,
        rollout_context_length=5,
        do_sample=False,
    )
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 20, 2]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 20, 2]]),
    }
    assert generator.generate(_ImmediateEosModel(), inputs) == []


def test_student_rollout_stops_on_every_model_declared_eos() -> None:
    class AlternateEosModel(torch.nn.Module):
        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            response = torch.tensor([[7, 3, 8]], device=input_ids.device)
            return SimpleNamespace(sequences=torch.cat((input_ids, response), dim=1))

    generator = StudentRolloutGenerator(
        tokenizer=_Tokenizer(),
        cutoff_len=8,
        rollout_context_length=5,
        do_sample=False,
        eos_token_ids=[2, 3],
    )
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 20, 2]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 20, 2]]),
    }

    features = generator.generate(AlternateEosModel(), inputs)

    assert generator.generation_config.eos_token_id == [2, 3]
    assert features[0]["input_ids"].tolist() == [10, 11, 12, 7]
