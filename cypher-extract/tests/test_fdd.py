import pytest
import torch
import torch.nn.functional as F

from distillation.fdd import _masked_mean, causal_hidden_state_mask, causal_response_mask, fdd_loss


def _template_fdd_loss(student_hidden, teacher_hidden, mask, student_head, teacher_head, student_map, teacher_map):
    trajectory_loss = 0.0
    derivative_loss = 0.0
    previous_student_logs = None
    previous_teacher_logs = None
    for index, (student_index, teacher_index) in enumerate(zip(student_map, teacher_map, strict=True)):
        student_logits = student_head(student_hidden[student_index])
        teacher_logits = teacher_head(teacher_hidden[teacher_index])
        student_probs = F.log_softmax(student_logits / 2.0, dim=-1)
        teacher_probs = F.softmax(teacher_logits / 2.0, dim=-1)
        token_loss = F.kl_div(student_probs, teacher_probs, reduction="none").sum(dim=-1)
        trajectory_loss = trajectory_loss + (token_loss * mask).sum() / mask.sum()

        student_logs = F.log_softmax(student_logits, dim=-1)
        teacher_logs = F.log_softmax(teacher_logits, dim=-1)
        if index > 0:
            cosine_loss = 1.0 - F.cosine_similarity(
                student_logs - previous_student_logs,
                teacher_logs - previous_teacher_logs,
                dim=-1,
                eps=1e-5,
            )
            derivative_loss = derivative_loss + (cosine_loss * mask).sum() / mask.sum()
        previous_student_logs = student_logs
        previous_teacher_logs = teacher_logs

    return trajectory_loss / len(student_map) + derivative_loss / (len(student_map) - 1)


def test_fdd_is_finite_and_backpropagates_to_student() -> None:
    generator = torch.Generator().manual_seed(11)
    student_hidden = tuple(torch.randn(2, 4, 5, generator=generator, requires_grad=True) for _ in range(4))
    teacher_hidden = tuple(torch.randn(2, 4, 7, generator=generator) for _ in range(5))
    student_head = torch.nn.Linear(5, 13, bias=False)
    teacher_head = torch.nn.Linear(7, 13, bias=False)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    loss = fdd_loss(student_hidden, teacher_hidden, mask, student_head, teacher_head, [1, 3], [2, 4])
    assert torch.isfinite(loss)
    loss.backward()
    assert student_hidden[1].grad is not None
    assert student_hidden[3].grad is not None
    assert all(hidden.grad is None for hidden in teacher_hidden)
    assert teacher_head.weight.grad is None


def test_fdd_projects_only_response_hidden_states() -> None:
    observed_shapes: list[tuple[int, ...]] = []

    class RecordingLinear(torch.nn.Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            observed_shapes.append(tuple(inputs.shape))
            return super().forward(inputs)

    generator = torch.Generator().manual_seed(13)
    student_hidden = tuple(torch.randn(2, 4, 5, generator=generator) for _ in range(3))
    teacher_hidden = tuple(torch.randn(2, 4, 7, generator=generator) for _ in range(3))
    student_head = RecordingLinear(5, 11, bias=False)
    teacher_head = RecordingLinear(7, 11, bias=False)
    response_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]])

    loss = fdd_loss(
        student_hidden,
        teacher_hidden,
        response_mask,
        student_head,
        teacher_head,
        [1, 2],
        [1, 2],
    )

    assert torch.isfinite(loss)
    assert observed_shapes == [(3, 5), (3, 7), (3, 5), (3, 7)]


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_masked_mean_neutralizes_ignored_non_finite_values(non_finite: float) -> None:
    values = torch.tensor([[non_finite, 2.0]], requires_grad=True)
    mask = torch.tensor([[0, 1]])

    loss = _masked_mean(values, mask)

    torch.testing.assert_close(loss, torch.tensor(2.0))
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 1.0]]))


def test_fdd_aligns_different_padded_vocab_sizes() -> None:
    generator = torch.Generator().manual_seed(19)
    student_hidden = tuple(torch.randn(1, 3, 4, generator=generator, requires_grad=True) for _ in range(3))
    teacher_hidden = tuple(torch.randn(1, 3, 6, generator=generator) for _ in range(3))
    student_head = torch.nn.Linear(4, 7, bias=False)
    teacher_head = torch.nn.Linear(6, 9, bias=False)

    actual = fdd_loss(
        student_hidden,
        teacher_hidden,
        torch.ones(1, 3),
        student_head,
        teacher_head,
        [1, 2],
        [1, 2],
    )
    actual.backward()

    assert torch.isfinite(actual)
    assert student_hidden[1].grad is not None


def test_fdd_rejects_single_layer_mapping() -> None:
    hidden = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
    head = torch.nn.Linear(3, 4, bias=False)
    with pytest.raises(ValueError, match="at least two"):
        fdd_loss(hidden, hidden, torch.ones(1, 2), head, head, [1], [1])


def test_causal_hidden_state_mask_removes_last_valid_token_per_row() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
    expected = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
    torch.testing.assert_close(causal_hidden_state_mask(attention_mask), expected)


def test_causal_response_mask_keeps_only_assistant_and_function_targets() -> None:
    # Positions supervise their *next* token. The three non-ignored targets
    # stand for a function call's two tokens and the final assistant token;
    # tool-observation tokens remain ignored context.
    labels = torch.tensor([[-100, -100, 10, 11, -100, 20, -100]])
    attention_mask = torch.ones_like(labels)
    expected = torch.tensor([[0, 1, 1, 0, 1, 0, 0]])
    torch.testing.assert_close(causal_response_mask(labels, attention_mask), expected)


def test_fdd_bf16_matches_template_formula() -> None:
    generator = torch.Generator().manual_seed(29)
    student_hidden = tuple(torch.randn(2, 3, 4, generator=generator, dtype=torch.bfloat16) for _ in range(4))
    teacher_hidden = tuple(torch.randn(2, 3, 6, generator=generator, dtype=torch.bfloat16) for _ in range(5))
    student_head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    teacher_head = torch.nn.Linear(6, 8, bias=False, dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bfloat16)
    student_map = [1, 3]
    teacher_map = [2, 4]

    actual = fdd_loss(
        student_hidden,
        teacher_hidden,
        mask,
        student_head,
        teacher_head,
        student_map,
        teacher_map,
    )
    expected = _template_fdd_loss(
        student_hidden,
        teacher_hidden,
        mask,
        student_head,
        teacher_head,
        student_map,
        teacher_map,
    )
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
