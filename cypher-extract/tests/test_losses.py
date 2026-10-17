from __future__ import annotations

import pytest
import torch

from distillation.losses import (
    LOSS_FUNCTIONS,
    _masked_token_mean,
    _sanitize_logits,
    align_causal_logits_and_labels,
    causal_lm_loss,
    compute_distillation_loss,
    compute_hpd_loss,
    forward_kl,
    reverse_kl,
)


def test_causal_lm_loss_matches_huggingface_token_mean() -> None:
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 5, 7, generator=generator)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, -100, 6]])

    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(causal_lm_loss(logits, labels), expected)


def test_causal_lm_loss_upcasts_bf16_logits() -> None:
    logits = torch.randn(2, 4, 7, dtype=torch.bfloat16)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])
    loss = causal_lm_loss(logits, labels)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_sanitize_logits_returns_finite_input_without_copying() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)

    sanitized = _sanitize_logits(logits)

    assert sanitized is logits


def test_causal_alignment_compacts_to_response_tokens_before_sanitizing() -> None:
    student = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
    teacher = student + 1.0
    student[0, 0, 0] = float("nan")
    labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, -100, 4]])

    aligned_student, aligned_teacher, aligned_labels = align_causal_logits_and_labels(
        student,
        teacher,
        labels,
    )

    assert aligned_student.shape == (4, 5)
    assert aligned_teacher.shape == (4, 5)
    assert aligned_labels.tolist() == [1, 2, 3, 4]
    assert torch.isfinite(aligned_student).all()


@pytest.mark.parametrize("method", ["fkl", "rkl", "sfkl", "srkl", "bdl", "csd", "amid"])
def test_response_token_compaction_preserves_finite_loss_and_gradients(method: str) -> None:
    generator = torch.Generator().manual_seed(67)
    student_data = torch.randn(2, 5, 11, generator=generator)
    teacher = torch.randn(2, 5, 11, generator=generator)
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 6, 7, -100, 8]])
    compact_student = student_data.clone().requires_grad_()
    full_student = student_data.clone().requires_grad_()

    compact_loss = compute_distillation_loss(method, compact_student, teacher, labels)
    shifted_labels = labels[:, 1:]
    if method in {"sfkl", "srkl"}:
        full_loss = LOSS_FUNCTIONS[method](full_student[:, :-1], teacher[:, :-1], shifted_labels, alpha=0.1)
    elif method == "bdl":
        full_loss = LOSS_FUNCTIONS[method](full_student[:, :-1], teacher[:, :-1], shifted_labels, lam=0.9)
    else:
        full_loss = LOSS_FUNCTIONS[method](full_student[:, :-1], teacher[:, :-1], shifted_labels)

    compact_loss.backward()
    full_loss.backward()

    torch.testing.assert_close(compact_loss, full_loss)
    torch.testing.assert_close(compact_student.grad, full_student.grad)


@pytest.mark.parametrize(
    "method",
    [
        "fkl",
        "rkl",
        "sfkl",
        "srkl",
        "csd",
    ],
)
def test_core_losses_are_finite_and_mask_non_target_tokens(method: str) -> None:
    generator = torch.Generator().manual_seed(7)
    student = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 11, generator=generator)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5], [-100, 6, 7, -100, 8]],
        dtype=torch.long,
    )

    actual = compute_distillation_loss(method, student, teacher, labels, skew_alpha=0.1)
    actual.backward()
    assert torch.isfinite(actual)
    assert student.grad is not None
    # Row zero's logits at position zero predict labels[0, 1], which is masked.
    torch.testing.assert_close(student.grad[0, 0], torch.zeros_like(student.grad[0, 0]))


def test_csd_bf16_is_finite() -> None:
    generator = torch.Generator().manual_seed(17)
    student = torch.randn(2, 4, 9, generator=generator, dtype=torch.bfloat16)
    teacher = torch.randn(2, 4, 9, generator=generator, dtype=torch.bfloat16)
    labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, 4, 5]])
    actual = compute_distillation_loss("csd", student, teacher, labels)
    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual)


def test_forward_kl_retains_teacher_entropy_constant() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    labels = torch.tensor([[1]])
    loss = forward_kl(logits, logits.detach(), labels)
    assert loss.item() > 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_reverse_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    labels = torch.tensor([[1]])
    torch.testing.assert_close(reverse_kl(logits, logits, labels), torch.tensor(0.0))


@pytest.mark.parametrize("method", ["sfkl", "srkl"])
def test_skewed_kl_is_finite_when_softmax_probability_underflows(method: str) -> None:
    student = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, -300.0], [0.0, 0.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    teacher = student.detach().clone()
    labels = torch.tensor([[-100, 1]])

    loss = compute_distillation_loss(method, student, teacher, labels)

    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize("method", ["fkl", "rkl", "sfkl", "srkl", "bdl", "csd", "amid", "hpd"])
@pytest.mark.parametrize("owner", ["student", "teacher"])
@pytest.mark.parametrize("non_finite", [float("-inf"), float("inf"), float("nan")])
def test_distillation_sanitizes_non_finite_logits_before_probability_operations(
    method: str,
    owner: str,
    non_finite: float,
) -> None:
    generator = torch.Generator().manual_seed(53)
    student_data = torch.randn(1, 4, 7, generator=generator)
    teacher = torch.randn(1, 4, 7, generator=generator)
    if owner == "student":
        student_data[0, 1, -1] = non_finite
    else:
        teacher[0, 1, -1] = non_finite
    student = student_data.requires_grad_()
    labels = torch.tensor([[-100, -100, 1, 2]])

    torch.manual_seed(59)
    loss = compute_distillation_loss(method, student, teacher, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_masked_token_mean_neutralizes_ignored_non_finite_values(non_finite: float) -> None:
    values = torch.tensor([[non_finite, 2.0]], requires_grad=True)
    labels = torch.tensor([[-100, 1]])

    loss = _masked_token_mean(values, labels)

    torch.testing.assert_close(loss, torch.tensor(2.0))
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 1.0]]))


def test_causal_alignment_uses_shifted_response_mask() -> None:
    student = torch.zeros(1, 4, 3)
    teacher = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    baseline = compute_distillation_loss("rkl", student, teacher, labels)

    changed_masked_position = student.clone()
    changed_masked_position[:, 0, 0] = 100.0
    masked = compute_distillation_loss("rkl", changed_masked_position, teacher, labels)
    torch.testing.assert_close(masked, baseline)

    changed_response_position = student.clone()
    changed_response_position[:, 1, 0] = 100.0
    unmasked = compute_distillation_loss("rkl", changed_response_position, teacher, labels)
    assert unmasked > baseline


@pytest.mark.parametrize("method", ["fkl", "rkl", "sfkl", "srkl", "csd", "bdl", "amid"])
def test_distillation_aligns_different_padded_vocab_sizes(method: str) -> None:
    generator = torch.Generator().manual_seed(41)
    student = torch.randn(1, 4, 7, generator=generator, requires_grad=True)
    teacher = torch.randn(1, 4, 9, generator=generator)
    labels = torch.tensor([[-100, -100, 3, 8]])

    actual = compute_distillation_loss(method, student, teacher, labels)
    actual.backward()

    assert torch.isfinite(actual)
    assert student.grad is not None
    # Token 8 exists only in the teacher's padded output vocabulary and is
    # excluded from the shared-vocabulary supervision mask.
    torch.testing.assert_close(student.grad[:, 2], torch.zeros_like(student.grad[:, 2]))


def test_empty_response_mask_is_rejected() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.full((1, 3), -100)
    with pytest.raises(ValueError, match="no response tokens"):
        compute_distillation_loss("fkl", logits, logits, labels)


def test_hpd_loss_is_finite_and_returns_metrics() -> None:
    generator = torch.Generator().manual_seed(31)
    student = torch.randn(2, 6, 13, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 6, 13, generator=generator)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5, 6], [-100, 2, 7, -100, 8, 9]],
        dtype=torch.long,
    )

    torch.manual_seed(11)
    actual, metrics = compute_hpd_loss(student, teacher, labels)
    assert torch.isfinite(actual)
    assert {"hpd_loss", "ce_loss", "sampled_diff_pct", "k1_gt_raw_mean"}.issubset(metrics)
    actual.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_hpd_samples_only_at_response_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, tuple[int, ...]] = {}

    def deterministic_multinomial(
        probabilities: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        observed["shape"] = tuple(probabilities.shape)
        return probabilities.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", deterministic_multinomial)
    generator = torch.Generator().manual_seed(71)
    student = torch.randn(2, 5, 9, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 9, generator=generator)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, -100, -100, 5]])

    loss, _ = compute_hpd_loss(student, teacher, labels)

    assert torch.isfinite(loss)
    assert observed["shape"] == (5, 9)


def test_compute_distillation_loss_dispatches_to_hpd() -> None:
    generator = torch.Generator().manual_seed(37)
    student = torch.randn(1, 5, 9, generator=generator)
    teacher = torch.randn(1, 5, 9, generator=generator)
    labels = torch.tensor([[-100, -100, 1, 2, 3]])

    torch.manual_seed(13)
    direct, _ = compute_hpd_loss(student, teacher, labels)
    torch.manual_seed(13)
    dispatched = compute_distillation_loss("hpd", student, teacher, labels)
    torch.testing.assert_close(dispatched, direct)


def test_bdl_is_zero_for_identical_distributions() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, -1.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 1]])
    loss = compute_distillation_loss("bdl", logits, logits.detach(), labels, bdl_lambda=0.9)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)
    loss.backward()
    assert logits.grad is not None


def test_bdl_matches_bidirectional_mixture_definition() -> None:
    generator = torch.Generator().manual_seed(31)
    student = torch.randn(2, 4, 7, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 4, 7, generator=generator)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]])
    lam = 0.9

    actual = compute_distillation_loss("bdl", student, teacher, labels, bdl_lambda=lam)

    student_probs = torch.softmax(student[:, :-1].float(), dim=-1)
    teacher_probs = torch.softmax(teacher[:, :-1].float(), dim=-1)
    p_mix = (1.0 - lam) * teacher_probs + lam * student_probs
    q_mix = lam * teacher_probs + (1.0 - lam) * student_probs
    token_loss = (p_mix * (p_mix.log() - q_mix.log())).sum(dim=-1)
    response_mask = labels[:, 1:].ne(-100)
    expected = (token_loss * response_mask).sum() / response_mask.sum()

    torch.testing.assert_close(actual, expected)


def test_bdl_rejects_lambda_that_makes_the_loss_identically_zero() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[-100, 1]])
    with pytest.raises(ValueError, match="identical"):
        compute_distillation_loss("bdl", logits, logits, labels, bdl_lambda=0.5)


@pytest.mark.parametrize(
    ("div_name", "div_order"),
    [(div_name, div_order) for div_name in ("fkl", "ab") for div_order in ("pr", "qr", "rp", "rq")],
)
def test_amid_loss_and_gradient_are_finite(div_name: str, div_order: str) -> None:
    generator = torch.Generator().manual_seed(23)
    student = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 11, generator=generator)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5], [-100, 6, 7, -100, 8]],
        dtype=torch.long,
    )

    actual = compute_distillation_loss(
        "amid",
        student,
        teacher,
        labels,
        amid_div_name=div_name,
        amid_div_order=div_order,
        amid_alpha=0.5,
        amid_lam=0.5,
    )
    actual.backward()
    assert torch.isfinite(actual)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
