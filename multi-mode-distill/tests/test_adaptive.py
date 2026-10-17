import math

import pytest

from src.adaptive import AdaptiveConfig, AdaptiveScheduler


def test_all_mode_probabilities_preserve_existing_behavior():
    scheduler = AdaptiveScheduler(seed=2)

    assert scheduler.mode_probabilities == pytest.approx({
        "off_policy": 0.9,
        "self_distill": 0.05,
        "on_policy": 0.05,
    })
    # The original scheduler routes a high random draw to OFF.
    assert scheduler.sample_mode() == "off_policy"


def test_off_self_disables_on_policy_sampling_and_evaluation():
    scheduler = AdaptiveScheduler(seed=1, mode_set="off_self")

    assert scheduler.mode_probabilities == pytest.approx({
        "off_policy": 0.95,
        "self_distill": 0.05,
        "on_policy": 0.0,
    })
    assert {scheduler.sample_mode() for _ in range(200)} <= {
        "off_policy", "self_distill"
    }
    log = scheduler.on_evaluation(self_loss=1.0)
    assert log["scheduler/eval_on_loss"] is None
    assert log["scheduler/ref_on_loss"] is None


def test_on_self_normalizes_probabilities_and_updates_only_enabled_modes():
    scheduler = AdaptiveScheduler(seed=1, mode_set="on_self")

    assert scheduler.mode_probabilities == pytest.approx({
        "off_policy": 0.0,
        "self_distill": 0.5,
        "on_policy": 0.5,
    })
    assert {scheduler.sample_mode() for _ in range(200)} <= {
        "self_distill", "on_policy"
    }

    scheduler.on_evaluation(self_loss=1.0, on_loss=1.0)
    log = scheduler.on_evaluation(self_loss=2.0, on_loss=0.5)

    assert log["scheduler/self_updated"] is True
    assert log["scheduler/on_updated"] is False
    assert math.isclose(log["scheduler/rho_self"], 0.6)
    assert math.isclose(log["scheduler/rho_on"], 0.4)
    assert log["scheduler/rho_off"] == 0.0


def test_on_self_requires_a_positive_initial_weight():
    config = AdaptiveConfig(
        rho_self_init=0.0,
        rho_on_init=0.0,
        rho_self_max=0.25,
        rho_on_max=0.25,
    )

    with pytest.raises(ValueError, match="positive initial"):
        AdaptiveScheduler(config, mode_set="on_self")
