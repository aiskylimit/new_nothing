"""Behavioral tests for ``KDTrainer.compute_loss``.

The trainer subclasses a LlamaFactory trainer whose ``__init__`` needs a full
training stack, so these tests build the instance with ``__new__`` and attach
only the attributes ``compute_loss`` actually reads. That keeps the assertions
on real tensors and real loss values rather than on trainer source text.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from distillation.arguments import DistillationArguments
from distillation.fdd import causal_response_mask, fdd_loss
from distillation.generation import selector_generation_kwargs
from distillation.losses import causal_lm_loss, compute_distillation_loss
from distillation.task_balancing import GENERATOR_TASK_ID, SELECTOR_TASK_ID

try:
    from distillation.trainer import KDTrainer
except ImportError as exc:
    pytest.skip(f"KD trainer dependencies are unavailable: {exc}", allow_module_level=True)

VOCAB_SIZE = 11
HIDDEN_SIZE = 6
SEQUENCE_LENGTH = 7
PROMPT_LENGTH = 3
FDD_LAYER_MAPPING = [2, 3]


class _TinyCausalLM(torch.nn.Module):
    """Deterministic stand-in exposing only what ``compute_loss`` touches."""

    def __init__(self, *, seed: int, num_blocks: int = 3) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False) for _ in range(num_blocks)
        )
        self.head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
        self.calls: list[dict[str, bool]] = []

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        self.calls.append(
            {
                "has_labels": labels is not None,
                "has_attention_mask": attention_mask is not None,
                "output_hidden_states": bool(output_hidden_states),
            }
        )
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        for block in self.blocks:
            hidden = torch.tanh(block(hidden))
            hidden_states.append(hidden)
        logits = self.head(hidden)
        return SimpleNamespace(
            loss=causal_lm_loss(logits, labels) if labels is not None else None,
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model


def _make_trainer(
    distillation_args: DistillationArguments,
    ref_model: torch.nn.Module | None = None,
) -> KDTrainer:
    trainer = KDTrainer.__new__(KDTrainer)
    trainer.distillation_args = distillation_args
    trainer.ref_model = ref_model
    trainer.accelerator = _Accelerator()
    trainer._stored_hpd_metrics = defaultdict(list)
    trainer._stored_loss_metrics = defaultdict(list)
    return trainer


def _batch(task_ids: list[int], *, seed: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB_SIZE, (len(task_ids), SEQUENCE_LENGTH), generator=generator)
    labels = input_ids.clone()
    labels[:, :PROMPT_LENGTH] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "task_ids": torch.tensor(task_ids, dtype=torch.long),
    }


def _student_and_teacher() -> tuple[_TinyCausalLM, _TinyCausalLM]:
    return _TinyCausalLM(seed=1).train(), _TinyCausalLM(seed=2).train()


def _forward(model: _TinyCausalLM, inputs: dict[str, torch.Tensor], **kwargs: object) -> SimpleNamespace:
    return model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **kwargs)


def test_sft_returns_the_student_lm_loss_without_a_teacher_forward() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(DistillationArguments(distill_method="sft", kd_ratio=0.0), ref_model=teacher)
    inputs = _batch([GENERATOR_TASK_ID, GENERATOR_TASK_ID])
    expected = _forward(student, inputs, labels=inputs["labels"]).loss

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert teacher.calls == []


def test_zero_kd_ratio_makes_a_kd_method_a_true_sft_run() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(DistillationArguments(distill_method="fkl", kd_ratio=0.0), ref_model=teacher)
    inputs = _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID])
    expected = _forward(student, inputs, labels=inputs["labels"]).loss

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert teacher.calls == []


def test_evaluation_reports_student_lm_loss_without_a_teacher_forward() -> None:
    student, teacher = _student_and_teacher()
    student.eval()
    trainer = _make_trainer(DistillationArguments(distill_method="fkl", kd_ratio=0.7), ref_model=teacher)
    inputs = _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID])
    expected = _forward(student, inputs, labels=inputs["labels"]).loss

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert teacher.calls == []
    assert trainer._stored_loss_metrics == {}


def test_kd_loss_mixes_lm_and_distillation_terms_by_kd_ratio() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(DistillationArguments(distill_method="fkl", kd_ratio=0.7), ref_model=teacher)
    inputs = _batch([GENERATOR_TASK_ID, GENERATOR_TASK_ID, SELECTOR_TASK_ID, SELECTOR_TASK_ID])
    student_outputs = _forward(student, inputs, labels=inputs["labels"])
    with torch.no_grad():
        teacher_outputs = _forward(teacher, inputs)
    expected_kd = compute_distillation_loss("fkl", student_outputs.logits, teacher_outputs.logits, inputs["labels"])
    expected = 0.3 * student_outputs.loss + 0.7 * expected_kd

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_the_teacher_forward_never_receives_supervised_labels() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(DistillationArguments(distill_method="fkl", kd_ratio=0.7), ref_model=teacher)

    trainer.compute_loss(student, _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID]))

    assert teacher.calls
    assert all(call["has_labels"] is False for call in teacher.calls)
    assert all(call["has_attention_mask"] is True for call in teacher.calls)


def test_kd_loss_backpropagates_into_the_student_but_not_the_teacher() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(DistillationArguments(distill_method="srkl", kd_ratio=0.5), ref_model=teacher)

    loss = trainer.compute_loss(student, _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID]))
    loss.backward()

    assert student.head.weight.grad is not None
    assert torch.isfinite(student.head.weight.grad).all()
    assert torch.any(student.head.weight.grad != 0)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_task_normalization_is_a_no_op_for_a_single_task_batch() -> None:
    student, teacher = _student_and_teacher()
    baseline = _make_trainer(DistillationArguments(distill_method="fkl", kd_ratio=0.5), ref_model=teacher)
    normalized = _make_trainer(
        DistillationArguments(distill_method="fkl", kd_ratio=0.5, selector_loss_weight=0.3),
        ref_model=teacher,
    )
    rows = [SELECTOR_TASK_ID, SELECTOR_TASK_ID, SELECTOR_TASK_ID]

    expected = baseline.compute_loss(student, _batch(rows))
    actual = normalized.compute_loss(student, _batch(rows))

    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_task_normalization_weights_tasks_independently_of_row_counts() -> None:
    student = _TinyCausalLM(seed=1).train()
    rows = [GENERATOR_TASK_ID, GENERATOR_TASK_ID, GENERATOR_TASK_ID, SELECTOR_TASK_ID]
    inputs = _batch(rows)
    logits = _forward(student, inputs).logits
    generator_mask = torch.tensor([True, True, True, False])
    selector_mask = torch.tensor([False, False, False, True])
    expected = 0.5 * causal_lm_loss(logits[generator_mask], inputs["labels"][generator_mask]) + 0.5 * causal_lm_loss(
        logits[selector_mask], inputs["labels"][selector_mask]
    )
    normalized = _make_trainer(DistillationArguments(distill_method="sft", kd_ratio=0.0, selector_loss_weight=0.5))
    plain = _make_trainer(DistillationArguments(distill_method="sft", kd_ratio=0.0))

    actual = normalized.compute_loss(student, _batch(rows))
    unweighted = plain.compute_loss(student, _batch(rows))

    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)
    # The single selector row carries half the loss, so token-count weighting
    # and task weighting must not agree on this batch.
    assert actual.item() != pytest.approx(unweighted.item(), rel=1e-4)


def test_task_normalized_loss_requires_task_ids_from_the_collator() -> None:
    student = _TinyCausalLM(seed=1).train()
    trainer = _make_trainer(DistillationArguments(distill_method="sft", kd_ratio=0.0, selector_loss_weight=0.5))
    inputs = _batch([GENERATOR_TASK_ID])
    inputs.pop("task_ids")

    with pytest.raises(ValueError, match="task_ids"):
        trainer.compute_loss(student, inputs)


def test_compute_loss_requires_labels() -> None:
    student = _TinyCausalLM(seed=1).train()
    trainer = _make_trainer(DistillationArguments(distill_method="sft", kd_ratio=0.0))
    inputs = _batch([GENERATOR_TASK_ID])
    inputs.pop("labels")

    with pytest.raises(ValueError, match="labels"):
        trainer.compute_loss(student, inputs)


def _fdd_arguments(fdd_weight: float) -> DistillationArguments:
    return DistillationArguments(
        distill_method="fdd_sfkl",
        kd_ratio=1.0,
        fdd_weight=fdd_weight,
        student_layer_mapping=list(FDD_LAYER_MAPPING),
        teacher_layer_mapping=list(FDD_LAYER_MAPPING),
    )


def test_fdd_weight_zero_keeps_token_distillation_only() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(_fdd_arguments(0.0), ref_model=teacher)
    inputs = _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID])
    student_outputs = _forward(student, inputs, output_hidden_states=True)
    with torch.no_grad():
        teacher_outputs = _forward(teacher, inputs, output_hidden_states=True)
    expected = 2.0 * compute_distillation_loss(
        "sfkl", student_outputs.logits, teacher_outputs.logits, inputs["labels"], skew_alpha=0.1
    )

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_fdd_weight_one_keeps_feature_distillation_only() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(_fdd_arguments(1.0), ref_model=teacher)
    inputs = _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID])
    student_outputs = _forward(student, inputs, output_hidden_states=True)
    with torch.no_grad():
        teacher_outputs = _forward(teacher, inputs, output_hidden_states=True)
    expected = 2.0 * fdd_loss(
        student_outputs.hidden_states,
        teacher_outputs.hidden_states,
        causal_response_mask(inputs["labels"], inputs["attention_mask"]),
        student.get_output_embeddings(),
        teacher.get_output_embeddings(),
        FDD_LAYER_MAPPING,
        FDD_LAYER_MAPPING,
    )

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_fdd_requests_hidden_states_from_both_models_during_training() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(_fdd_arguments(0.5), ref_model=teacher)

    trainer.compute_loss(student, _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID]))

    assert all(call["output_hidden_states"] is True for call in student.calls)
    assert all(call["output_hidden_states"] is True for call in teacher.calls)


def test_fdd_task_normalization_balances_token_and_feature_losses() -> None:
    student, teacher = _student_and_teacher()
    selector_weight = 0.4
    fdd_weight = 0.25
    trainer = _make_trainer(
        DistillationArguments(
            distill_method="fdd_sfkl",
            kd_ratio=1.0,
            selector_loss_weight=selector_weight,
            fdd_weight=fdd_weight,
            student_layer_mapping=list(FDD_LAYER_MAPPING),
            teacher_layer_mapping=list(FDD_LAYER_MAPPING),
        ),
        ref_model=teacher,
    )
    inputs = _batch([GENERATOR_TASK_ID, GENERATOR_TASK_ID, GENERATOR_TASK_ID, SELECTOR_TASK_ID])
    student_outputs = _forward(student, inputs, output_hidden_states=True)
    with torch.no_grad():
        teacher_outputs = _forward(teacher, inputs, output_hidden_states=True)
    feature_mask = causal_response_mask(inputs["labels"], inputs["attention_mask"])

    def task_components(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_loss = compute_distillation_loss(
            "sfkl",
            student_outputs.logits[mask],
            teacher_outputs.logits[mask],
            inputs["labels"][mask],
            skew_alpha=0.1,
        )
        feature_loss = fdd_loss(
            tuple(hidden[mask] for hidden in student_outputs.hidden_states),
            tuple(hidden[mask] for hidden in teacher_outputs.hidden_states),
            feature_mask[mask],
            student.get_output_embeddings(),
            teacher.get_output_embeddings(),
            FDD_LAYER_MAPPING,
            FDD_LAYER_MAPPING,
        )
        return token_loss, feature_loss

    generator_components = task_components(inputs["task_ids"].eq(GENERATOR_TASK_ID))
    selector_components = task_components(inputs["task_ids"].eq(SELECTOR_TASK_ID))
    token_loss = (1.0 - selector_weight) * generator_components[0] + selector_weight * selector_components[0]
    feature_loss = (1.0 - selector_weight) * generator_components[1] + selector_weight * selector_components[1]
    expected = 2.0 * ((1.0 - fdd_weight) * token_loss + fdd_weight * feature_loss)

    loss = trainer.compute_loss(student, inputs)

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_training_records_the_loss_components_it_logs() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(_fdd_arguments(0.5), ref_model=teacher)

    trainer.compute_loss(student, _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID]))

    assert set(trainer._stored_loss_metrics) == {
        "train_lm_loss",
        "train_distill_loss",
        "train_token_kd_loss",
        "train_feature_loss",
    }


def test_hpd_records_per_task_metrics_under_task_normalization() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(
        DistillationArguments(distill_method="hpd", kd_ratio=0.5, selector_loss_weight=0.4),
        ref_model=teacher,
    )

    torch.manual_seed(0)
    trainer.compute_loss(student, _batch([GENERATOR_TASK_ID, GENERATOR_TASK_ID, SELECTOR_TASK_ID]))

    recorded = set(trainer._stored_hpd_metrics)
    assert recorded
    assert any(key.startswith("generator_") for key in recorded)
    assert any(key.startswith("selector_") for key in recorded)


def test_hpd_without_task_normalization_records_unprefixed_metrics() -> None:
    student, teacher = _student_and_teacher()
    trainer = _make_trainer(DistillationArguments(distill_method="hpd", kd_ratio=0.5), ref_model=teacher)

    torch.manual_seed(0)
    trainer.compute_loss(student, _batch([GENERATOR_TASK_ID, SELECTOR_TASK_ID]))

    recorded = set(trainer._stored_hpd_metrics)
    assert recorded
    assert not any(key.startswith(("generator_", "selector_")) for key in recorded)


class _PredictionTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    _decoded = {
        (7, 8): "MATCH (n) RETURN n",
        (9,): '{"label": "YES"}',
    }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self._decoded[tuple(token_ids)]


def test_prediction_step_left_pads_prompts_splits_tasks_and_restores_row_order(monkeypatch) -> None:
    trainer = KDTrainer.__new__(KDTrainer)
    trainer.args = SimpleNamespace(predict_with_generate=True)
    trainer.processing_class = _PredictionTokenizer()
    trainer._prepare_inputs = lambda inputs: {key: value.clone() for key, value in inputs.items()}
    trainer.compute_loss = lambda model, inputs: torch.tensor(2.5)
    trainer.compute_loss_context_manager = nullcontext
    inputs = {
        "input_ids": torch.tensor([[0, 10, 11, 7, 8], [20, 21, 22, 9, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, -100, 7, 8], [-100, -100, -100, 9, -100]]),
    }
    calls: list[tuple[dict[str, torch.Tensor], dict[str, object]]] = []

    def fake_prediction_step(self, model, task_inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        del self, model, ignore_keys
        assert prediction_loss_only is False
        calls.append(({key: value.clone() for key, value in task_inputs.items()}, kwargs))
        if len(calls) == 1:
            return None, torch.tensor([[31, 32, 33]]), None
        return None, torch.tensor([[41, 42]]), None

    monkeypatch.setattr(KDTrainer.__mro__[1], "prediction_step", fake_prediction_step)

    loss, generated_tokens, labels = trainer.prediction_step(
        object(),
        inputs,
        prediction_loss_only=False,
        do_sample=True,
        max_new_tokens=64,
    )

    assert loss.item() == pytest.approx(2.5)
    torch.testing.assert_close(labels, inputs["labels"])
    torch.testing.assert_close(generated_tokens, torch.tensor([[31, 32, 33], [41, 42, 0]]))
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0]["input_ids"], torch.tensor([[0, 10, 11]]))
    torch.testing.assert_close(calls[0][0]["attention_mask"], torch.tensor([[0, 1, 1]]))
    torch.testing.assert_close(calls[1][0]["input_ids"], torch.tensor([[20, 21, 22]]))
    torch.testing.assert_close(calls[1][0]["attention_mask"], torch.tensor([[1, 1, 1]]))
    assert calls[0][1]["do_sample"] is True
    assert calls[0][1]["max_new_tokens"] == 64
    assert calls[1][1] == {"do_sample": True, "max_new_tokens": 64, **selector_generation_kwargs()}
