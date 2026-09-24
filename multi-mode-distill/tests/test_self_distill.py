import random

from src.self_distill import sample_context


def test_fixed_drop_ratio_removes_a_deterministic_suffix():
    steps = ["one", "two", "three", "four", "five"]

    assert sample_context(
        steps, " | ", 0.5, random.Random(1), fixed_drop_ratio=0.4
    ) == "one | two | three"


def test_fixed_drop_ratio_rounds_removed_steps_up():
    steps = ["one", "two", "three", "four", "five"]

    assert sample_context(
        steps, " | ", 0.5, random.Random(1), fixed_drop_ratio=0.25
    ) == "one | two | three"


def test_fixed_drop_ratio_supports_endpoint_values():
    steps = ["one", "two", "three"]

    assert sample_context(
        steps, " | ", 0.5, random.Random(1), fixed_drop_ratio=0.0
    ) == "one | two | three"
    assert sample_context(
        steps, " | ", 0.5, random.Random(1), fixed_drop_ratio=1.0
    ) == ""
    assert sample_context(
        ["one"], " | ", 0.5, random.Random(1), fixed_drop_ratio=0.0
    ) == "one"


def test_omitting_fixed_threshold_preserves_stochastic_sampling():
    steps = ["one", "two", "three", "four"]

    expected = sample_context(steps, " | ", 0.5, random.Random(7))
    actual = sample_context(
        steps, " | ", 0.5, random.Random(7), fixed_drop_ratio=None)

    assert actual == expected
