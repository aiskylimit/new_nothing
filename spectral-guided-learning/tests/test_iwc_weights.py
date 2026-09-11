import math

import pytest

from build_iwc_datasets import build_example
from iwc_weights import (
    reverse_iwc_step_weights,
    shuffled_iwc_step_weights,
    stable_iwc_step_weights,
    vanilla_iwc_step_weights,
)


def test_vanilla_iwc_has_unit_step_mean_but_not_token_mass_invariance():
    weights = vanilla_iwc_step_weights([0.0, 1.0], [1, 10], temperature=1.0)
    assert sum(weights) / 2 == pytest.approx(1.0)
    assert sum(length * weight for length, weight in zip([1, 10], weights)) != pytest.approx(11.0)


def test_stable_iwc_preserves_selected_token_mass():
    lengths = [1, 3, 11]
    weights = stable_iwc_step_weights([0.1, 0.7, 2.0], lengths, temperature=0.8, clip=2.0)
    assert sum(length * weight for length, weight in zip(lengths, weights)) == pytest.approx(sum(lengths))


def test_stable_iwc_lambda_zero_is_exactly_uniform():
    assert stable_iwc_step_weights([0.1, 3.0], [2, 9], interpolation=0.0) == [1.0, 1.0]


def test_stable_iwc_uniform_entropy_is_well_defined():
    assert stable_iwc_step_weights([2.0, 2.0, 2.0], [1, 2, 3]) == pytest.approx([1.0, 1.0, 1.0])


def test_iwc_builder_reuses_the_spectral_mask_and_keeps_stop_token_uniform():
    record = {
        "id": 7,
        "input_ids": list(range(7)),
        "response_token_span": [0, 6],
        "steps": [
            {"token_start": 0, "token_end": 2},
            {"token_start": 2, "token_end": 6},
        ],
    }
    example, stats = build_example(
        record,
        strengths=[3.0, 1.0],
        entropies=[0.0, 5.0],
        threshold=0.7,
        variant="iwc-stable",
        temperature=1.0,
        interpolation=0.0,
        clip=2.0,
        epsilon=1e-8,
    )

    # p=0.7 selects the first (two-token) step; lambda=0 must be spectral exactly.
    assert example["loss_mask"] == [1, 1, 0, 0, 0, 0, 1]
    assert example["loss_weights"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert stats["weight_mass_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_temperature", [0.0, -1.0])
def test_iwc_rejects_invalid_temperature(bad_temperature):
    with pytest.raises(ValueError):
        vanilla_iwc_step_weights([1.0], [1], bad_temperature)


def test_iwc_weights_are_positive_and_finite_after_clipping():
    weights = stable_iwc_step_weights([-1e9, 1e9], [5, 7], temperature=0.001, clip=1.0)
    assert all(math.isfinite(weight) and weight > 0 for weight in weights)


def test_shuffled_control_preserves_token_mass_but_changes_assignment():
    lengths = [1, 3, 11]
    entropies = [0.1, 0.7, 2.0]
    stable = stable_iwc_step_weights(entropies, lengths, temperature=0.8, clip=2.0)
    shuffled = shuffled_iwc_step_weights(entropies, lengths, temperature=0.8, clip=2.0, seed=1)

    assert sum(l * w for l, w in zip(lengths, shuffled)) == pytest.approx(sum(lengths))
    assert shuffled != stable  # same entropy set, different step assignment


def test_shuffled_control_is_deterministic_given_a_seed():
    lengths, entropies = [1, 3, 11], [0.1, 0.7, 2.0]
    first = shuffled_iwc_step_weights(entropies, lengths, seed=7)
    second = shuffled_iwc_step_weights(entropies, lengths, seed=7)
    assert first == second


def test_reverse_control_flips_the_entropy_to_weight_ranking():
    lengths = [4, 4, 4]
    entropies = [0.1, 1.0, 5.0]  # strictly increasing
    stable = stable_iwc_step_weights(entropies, lengths, temperature=1.0)
    reversed_weights = reverse_iwc_step_weights(entropies, lengths, temperature=1.0)

    assert stable[0] < stable[1] < stable[2]
    assert reversed_weights[0] > reversed_weights[1] > reversed_weights[2]
    assert sum(l * w for l, w in zip(lengths, reversed_weights)) == pytest.approx(sum(lengths))


def test_build_example_supports_lambda0_shuffled_and_reverse_variants():
    record = {
        "id": 3,
        "input_ids": list(range(7)),
        "response_token_span": [0, 6],
        "steps": [
            {"token_start": 0, "token_end": 2},
            {"token_start": 2, "token_end": 6},
        ],
    }
    for variant in ("iwc-stable-lambda0", "iwc-stable-shuffled", "iwc-stable-reverse"):
        example, stats = build_example(
            record, [3.0, 1.0], [0.0, 5.0], 0.7, variant, 1.0, 1.0, 2.0, 1e-8, seed=42,
        )
        assert example["loss_mask"] == [1, 1, 0, 0, 0, 0, 1]
        assert stats["weight_mass_ratio"] == pytest.approx(1.0)


def test_stable_dataset_preserves_total_weighted_mass_including_stop_token():
    record = {
        "id": 8,
        "input_ids": list(range(7)),
        "response_token_span": [0, 6],
        "steps": [
            {"token_start": 0, "token_end": 2},
            {"token_start": 2, "token_end": 6},
        ],
    }
    example, _ = build_example(
        record, [3.0, 2.0], [0.0, 3.0], 1.0, "iwc-stable", 1.0, 1.0, 2.0, 1e-8
    )
    weighted_mass = sum(
        weight for keep, weight in zip(example["loss_mask"], example["loss_weights"]) if keep
    )
    assert weighted_mass == pytest.approx(sum(example["loss_mask"]))
