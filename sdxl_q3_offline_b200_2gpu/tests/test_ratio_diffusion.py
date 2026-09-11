"""CPU-only mathematical regression tests for :mod:`ratio_diffusion`.

These tests intentionally use independently written Gaussian/DDPM equations.
They are meant to protect the transition-ratio implementation from seemingly
small coefficient or precision changes that alter the training objective.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from ratio_diffusion import (
    StepConfidenceHead,
    auxiliary_output_gradient_scale,
    ddpm_posterior_coefficients,
    dspo_winner_score_anchor,
    diffusion_dpo_loss_from_per_sample_losses,
    equal_covariance_gaussian_kl,
    expected_transition_log_uplift,
    loss_output_gradient,
    loss_output_gradient_norm,
    latent_teacher_preference_probability,
    mode_seeking_exponent_clip_fraction,
    mode_seeking_margin_gradient_magnitude,
    mode_seeking_power_loss_from_log_ratio,
    preference_loss_margin_scale,
    predict_clean_from_epsilon,
    reduce_latent,
    sample_true_posterior_action,
    scaled_basu_loss_from_log_ratio,
    step_aware_preference_loss,
    step_preference_consistency_target,
    select_timestep_pair_scores,
    symmetric_reference_kl_anchor,
    transition_log_uplift,
    transition_mean,
    true_posterior_mean,
    winner_denoising_anchor,
)


def _standard_ddpm_posterior(
    alphas_cumprod: torch.Tensor, timesteps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independently compute q(x_{t-1}|x_t,x_0)'s A, B, and variance."""
    alpha_bar_t = alphas_cumprod[timesteps]
    alpha_bar_previous = alphas_cumprod[timesteps - 1]
    alpha_step = alpha_bar_t / alpha_bar_previous
    beta_step = 1.0 - alpha_step

    coef_xt = alpha_step.sqrt() * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
    coef_x0 = alpha_bar_previous.sqrt() * beta_step / (1.0 - alpha_bar_t)
    variance = beta_step * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
    return coef_xt, coef_x0, variance


def _per_example(terms: torch.Tensor) -> torch.Tensor:
    return terms.flatten(start_dim=1)[:, 0]


def test_ddpm_posterior_coefficients_match_standard_formula() -> None:
    alphas_cumprod = torch.tensor([0.998, 0.944, 0.847, 0.716, 0.562, 0.401])
    timesteps = torch.tensor([1, 3, 5])
    sample_shape = torch.Size((3, 2, 4, 5))

    coefficients = ddpm_posterior_coefficients(alphas_cumprod, timesteps, sample_shape)
    expected_xt, expected_x0, expected_variance = _standard_ddpm_posterior(
        alphas_cumprod.double(), timesteps
    )

    torch.testing.assert_close(
        coefficients.coef_xt[:, 0, 0, 0], expected_xt.float(), rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        coefficients.coef_x0[:, 0, 0, 0], expected_x0.float(), rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        coefficients.variance[:, 0, 0, 0], expected_variance.float(), rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        coefficients.alpha_t[:, 0, 0, 0], alphas_cumprod[timesteps].sqrt()
    )
    torch.testing.assert_close(
        coefficients.sigma_t[:, 0, 0, 0], (1.0 - alphas_cumprod[timesteps]).sqrt()
    )


def test_sampled_true_posterior_actions_have_expected_moments() -> None:
    alphas_cumprod = torch.tensor([0.99, 0.92, 0.79, 0.63, 0.46, 0.31])
    timesteps = torch.tensor([1, 3, 5])
    # A large feature axis makes this a low-cost Monte-Carlo moment check.
    sample_shape = torch.Size((3, 65_536))
    coefficients = ddpm_posterior_coefficients(alphas_cumprod, timesteps, sample_shape)

    noisy_latents = torch.tensor([[-0.8], [0.25], [1.15]]).expand(sample_shape)
    clean_latents = torch.tensor([[0.6], [-1.3], [0.35]]).expand(sample_shape)
    posterior_noise = torch.randn(sample_shape, generator=torch.Generator().manual_seed(17))

    action, returned_mean = sample_true_posterior_action(
        noisy_latents, clean_latents, posterior_noise, coefficients
    )
    expected_mean = true_posterior_mean(noisy_latents, clean_latents, coefficients)
    centered = action - expected_mean

    torch.testing.assert_close(returned_mean, expected_mean)
    torch.testing.assert_close(centered.mean(dim=1), torch.zeros(3), atol=6e-3, rtol=0)
    torch.testing.assert_close(
        centered.square().mean(dim=1),
        _per_example(coefficients.variance),
        rtol=0.025,
        atol=2.5e-4,
    )


def test_step_confidence_head_initializes_to_uninformative_half_probability() -> None:
    head = StepConfidenceHead(
        feature_channels=8, timestep_embedding_dim=16, hidden_dim=24
    )
    features = torch.randn(
        (4, 8, 5, 7), generator=torch.Generator().manual_seed(23)
    )
    confidence = head(
        features,
        pair_timesteps=torch.tensor([1, 999]),
        num_train_timesteps=1000,
    )

    assert confidence.shape == (2,)
    torch.testing.assert_close(confidence, torch.full((2,), 0.5))
    assert sum(parameter.numel() for parameter in head.parameters()) < 10_000


def test_step_consistency_target_uses_reference_pair_order_and_smoothing() -> None:
    # Preferred samples are the first half. Pair 0 is locally consistent,
    # pair 1 is inconsistent, and pair 2 is an exact tie.
    reference_losses = torch.tensor([0.2, 0.8, 0.4, 0.6, 0.1, 0.4])
    target = step_preference_consistency_target(reference_losses)
    smoothed = step_preference_consistency_target(
        reference_losses, label_smoothing=0.1
    )

    torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 0.5]))
    torch.testing.assert_close(smoothed, torch.tensor([0.95, 0.05, 0.5]))


def test_precomputed_lrm_scores_select_the_sampled_timestep_bin() -> None:
    preferred = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]]
    )
    rejected = preferred - 0.5
    selected_w, selected_l = select_timestep_pair_scores(
        preferred,
        rejected,
        timesteps=torch.tensor([0, 500, 999]),
        num_train_timesteps=1000,
    )

    torch.testing.assert_close(selected_w, torch.tensor([10.0, 22.0, 33.0]))
    torch.testing.assert_close(selected_l, torch.tensor([9.5, 21.5, 32.5]))

    scalar_w, scalar_l = select_timestep_pair_scores(
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0, 3.0]),
        timesteps=torch.tensor([1, 999]),
        num_train_timesteps=1000,
    )
    torch.testing.assert_close(scalar_w, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(scalar_l, torch.tensor([0.0, 3.0]))


def test_lrm_score_gap_has_correct_bradley_terry_orientation() -> None:
    probability = latent_teacher_preference_probability(
        preferred_scores=torch.tensor([2.0, 0.0, 1.0]),
        rejected_scores=torch.tensor([0.0, 2.0, 1.0]),
        temperature=2.0,
    )
    torch.testing.assert_close(
        probability,
        torch.sigmoid(torch.tensor([1.0, -1.0, 0.0])),
    )
    assert probability[0] > 0.5
    assert probability[1] < 0.5
    torch.testing.assert_close(probability[2], torch.tensor(0.5))


def test_step_aware_policy_weight_is_positive_and_detached_from_policy_loss() -> None:
    per_pair_loss = torch.tensor([2.0, 4.0], requires_grad=True)
    confidence = torch.tensor([0.2, 0.8], requires_grad=True)
    target = torch.tensor([0.0, 1.0])
    output = step_aware_preference_loss(
        per_pair_loss,
        confidence,
        target,
        confidence_loss_weight=0.0,
        minimum_policy_weight=0.1,
    )
    expected_weights = torch.tensor([0.28, 0.82])
    expected_policy_loss = (expected_weights * per_pair_loss.detach()).mean()

    torch.testing.assert_close(output.policy_weights, expected_weights)
    torch.testing.assert_close(output.policy_loss, expected_policy_loss)
    output.loss.backward()
    torch.testing.assert_close(per_pair_loss.grad, expected_weights / 2.0)
    # With BCE disabled, the weighted policy term must not teach the head to
    # lower its own weights; this is the anti-collapse stop-gradient.
    torch.testing.assert_close(confidence.grad, torch.zeros_like(confidence))


def test_step_aware_confidence_bce_trains_the_confidence_probability() -> None:
    per_pair_loss = torch.tensor([1.0, 1.0], requires_grad=True)
    confidence = torch.tensor([0.2, 0.8], requires_grad=True)
    output = step_aware_preference_loss(
        per_pair_loss,
        confidence,
        confidence_target=torch.tensor([1.0, 0.0]),
        confidence_loss_weight=2.0,
        minimum_policy_weight=0.05,
    )
    output.loss.backward()

    assert confidence.grad is not None
    assert confidence.grad[0] < 0
    assert confidence.grad[1] > 0


def test_step_aware_policy_weights_use_explicit_detached_mean_normalizer() -> None:
    per_pair_loss = torch.tensor([2.0, 4.0], requires_grad=True)
    confidence = torch.tensor([0.2, 0.8], requires_grad=True)
    normalizer = torch.tensor(0.55, requires_grad=True)
    output = step_aware_preference_loss(
        per_pair_loss,
        confidence,
        confidence_target=torch.tensor([0.0, 1.0]),
        confidence_loss_weight=0.0,
        minimum_policy_weight=0.1,
        policy_weight_normalizer=normalizer,
    )

    expected_raw_weights = torch.tensor([0.28, 0.82])
    expected_policy_weights = expected_raw_weights / expected_raw_weights.mean()
    torch.testing.assert_close(output.raw_policy_weights, expected_raw_weights)
    torch.testing.assert_close(output.policy_weights, expected_policy_weights)
    torch.testing.assert_close(output.policy_weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(output.policy_weight_normalizer, torch.tensor(0.55))
    torch.testing.assert_close(
        output.policy_loss,
        (expected_policy_weights * per_pair_loss.detach()).mean(),
    )

    output.loss.backward()
    torch.testing.assert_close(per_pair_loss.grad, expected_policy_weights / 2.0)
    # Both the supplied batch statistic and the confidence-derived policy
    # weights are stop-gradient controls for the policy objective.
    assert normalizer.grad is None
    torch.testing.assert_close(confidence.grad, torch.zeros_like(confidence))


def test_step_aware_no_normalizer_preserves_legacy_raw_weights() -> None:
    output = step_aware_preference_loss(
        per_pair_loss=torch.tensor([2.0, 4.0]),
        confidence=torch.tensor([0.2, 0.8]),
        confidence_target=torch.tensor([0.0, 1.0]),
        confidence_loss_weight=0.0,
        minimum_policy_weight=0.1,
    )

    expected_raw_weights = torch.tensor([0.28, 0.82])
    torch.testing.assert_close(output.raw_policy_weights, expected_raw_weights)
    torch.testing.assert_close(output.policy_weights, expected_raw_weights)
    torch.testing.assert_close(output.policy_weight_normalizer, torch.tensor(1.0))
    torch.testing.assert_close(
        output.policy_loss,
        (expected_raw_weights * torch.tensor([2.0, 4.0])).mean(),
    )


def test_step_aware_policy_can_use_detached_frozen_lrm_signal() -> None:
    per_pair_loss = torch.tensor([2.0, 4.0], requires_grad=True)
    head_confidence = torch.tensor([0.9, 0.1], requires_grad=True)
    lrm_probability = torch.tensor([0.2, 0.8], requires_grad=True)
    output = step_aware_preference_loss(
        per_pair_loss=per_pair_loss,
        confidence=head_confidence,
        confidence_target=lrm_probability.detach(),
        confidence_loss_weight=0.0,
        minimum_policy_weight=0.1,
        policy_confidence=lrm_probability,
    )

    expected_weights = torch.tensor([0.28, 0.82])
    torch.testing.assert_close(output.policy_weights, expected_weights)
    output.loss.backward()
    torch.testing.assert_close(per_pair_loss.grad, expected_weights / 2.0)
    torch.testing.assert_close(head_confidence.grad, torch.zeros_like(head_confidence))
    assert lrm_probability.grad is None


def test_preference_margin_scale_preserves_origin_gradient_across_beta() -> None:
    # For lambda_B=0 and c_B=4, d ell / d logR at the origin is 2/c_B=0.5.
    # The legacy beta=5000, scale=1 setup therefore has raw-margin gradient 2500.
    slope_at_origin = 0.5
    target_margin_gradient = 2_500.0
    legacy_margin = torch.zeros(1, requires_grad=True)
    legacy_loss = scaled_basu_loss_from_log_ratio(
        5_000.0 * legacy_margin,
        bregman_lambda=0.0,
        bregman_scale=4.0,
        max_exp_argument=None,
    ).sum()
    legacy_gradient = torch.autograd.grad(legacy_loss, legacy_margin)[0]

    calibrated_margin = torch.zeros(1, requires_grad=True)
    calibrated_scale = preference_loss_margin_scale(
        ratio_beta=500.0,
        loss_log_ratio_slope_at_zero=slope_at_origin,
        target_margin_gradient=target_margin_gradient,
    )
    calibrated_loss = calibrated_scale * scaled_basu_loss_from_log_ratio(
        500.0 * calibrated_margin,
        bregman_lambda=0.0,
        bregman_scale=4.0,
        max_exp_argument=None,
    ).sum()
    calibrated_gradient = torch.autograd.grad(calibrated_loss, calibrated_margin)[0]

    assert calibrated_scale == pytest.approx(10.0)
    torch.testing.assert_close(legacy_gradient, torch.tensor([target_margin_gradient]))
    torch.testing.assert_close(calibrated_gradient, legacy_gradient)


def test_power_bregman_pilot_matches_the_q2_origin_margin_gradient() -> None:
    margin = torch.zeros(1, requires_grad=True)
    scale = preference_loss_margin_scale(
        ratio_beta=500.0,
        loss_log_ratio_slope_at_zero=1.0,
        target_margin_gradient=2_500.0,
    )
    loss = scale * mode_seeking_power_loss_from_log_ratio(
        500.0 * margin,
        kappa=0.5,
        max_exp_argument=None,
    ).sum()
    gradient = torch.autograd.grad(loss, margin)[0]

    assert scale == pytest.approx(5.0)
    torch.testing.assert_close(gradient, torch.tensor([2_500.0]))


def test_symmetric_reference_anchor_is_mean_of_pairwise_kl_sums() -> None:
    # Preferred samples precede rejected samples, so pair sums are 1+2 and 4+8.
    per_sample_kl = torch.tensor([1.0, 4.0, 2.0, 8.0], requires_grad=True)
    anchor = symmetric_reference_kl_anchor(per_sample_kl)

    torch.testing.assert_close(anchor, torch.tensor(7.5))
    anchor.backward()
    torch.testing.assert_close(per_sample_kl.grad, torch.full((4,), 0.5))
    torch.testing.assert_close(
        symmetric_reference_kl_anchor(per_sample_kl.detach().flip(0)),
        anchor.detach(),
    )


def test_winner_denoising_anchor_uses_only_preferred_half() -> None:
    model_prediction = torch.tensor(
        [1.0, 3.0, 100.0, -100.0], requires_grad=True
    ).reshape(4, 1, 1, 1)
    model_prediction.retain_grad()
    target = torch.zeros_like(model_prediction)
    anchor = winner_denoising_anchor(model_prediction, target)

    torch.testing.assert_close(anchor, torch.tensor(5.0))
    anchor.backward()
    assert model_prediction.grad is not None
    torch.testing.assert_close(
        model_prediction.grad.flatten(), torch.tensor([1.0, 3.0, 0.0, 0.0])
    )

    changed_losers = model_prediction.detach().clone()
    changed_losers[2:] = torch.tensor([[[[1_000.0]]], [[[-2_000.0]]]])
    torch.testing.assert_close(
        winner_denoising_anchor(changed_losers, target), anchor.detach()
    )


def test_dspo_anchor_reduces_to_winner_mse_at_reference_initialization() -> None:
    model_prediction = torch.tensor(
        [1.0, 3.0, 100.0, -100.0], requires_grad=True
    ).reshape(4, 1, 1, 1)
    reference_prediction = model_prediction.detach().clone()
    target = torch.zeros_like(model_prediction)
    preferred_probability = torch.tensor([0.1, 0.9])

    dspo_anchor = dspo_winner_score_anchor(
        model_prediction=model_prediction,
        reference_prediction=reference_prediction,
        target=target,
        preferred_probability=preferred_probability,
        correction_scale=0.5,
    )
    winner_mse = winner_denoising_anchor(model_prediction, target)
    torch.testing.assert_close(dspo_anchor, winner_mse)


def test_dspo_anchor_matches_detached_formula_and_only_regresses_winner() -> None:
    model_prediction = torch.tensor(
        [2.0, 4.0, 100.0, -100.0], requires_grad=True
    ).reshape(4, 1, 1, 1)
    model_prediction.retain_grad()
    reference_prediction = torch.tensor([1.0, 1.0, -5.0, 8.0]).reshape(4, 1, 1, 1)
    target = torch.zeros_like(model_prediction)
    preferred_probability = torch.tensor([0.75, 0.25], requires_grad=True)
    correction_scale = 0.5

    anchor = dspo_winner_score_anchor(
        model_prediction=model_prediction,
        reference_prediction=reference_prediction,
        target=target,
        preferred_probability=preferred_probability,
        correction_scale=correction_scale,
    )
    expected_residual = torch.tensor([1.875, 2.875])
    torch.testing.assert_close(anchor, expected_residual.square().mean())
    anchor.backward()

    expected_jacobian = torch.tensor([0.875, 0.625])
    torch.testing.assert_close(
        model_prediction.grad.flatten(),
        torch.tensor(
            [
                expected_residual[0] * expected_jacobian[0],
                expected_residual[1] * expected_jacobian[1],
                0.0,
                0.0,
            ]
        ),
    )
    assert preferred_probability.grad is None


def test_zero_dspo_correction_is_exact_winner_mse_control() -> None:
    generator = torch.Generator().manual_seed(91)
    model_prediction = torch.randn((4, 2, 3, 3), generator=generator)
    reference_prediction = torch.randn((4, 2, 3, 3), generator=generator)
    target = torch.randn((4, 2, 3, 3), generator=generator)
    probability = torch.tensor([0.2, 0.9])
    torch.testing.assert_close(
        dspo_winner_score_anchor(
            model_prediction,
            reference_prediction,
            target,
            probability,
            correction_scale=0.0,
        ),
        winner_denoising_anchor(model_prediction, target),
    )


def test_dspo_anchor_rejects_a_jacobian_reversing_correction() -> None:
    values = torch.zeros((2, 1, 1, 1))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        dspo_winner_score_anchor(
            values,
            values,
            values,
            torch.tensor([0.5]),
            correction_scale=1.0,
        )


def test_auxiliary_output_gradient_scale_matches_ratio_cap_and_zero_guards() -> None:
    primary = torch.tensor(4.0, requires_grad=True)
    auxiliary = torch.tensor(2.0, requires_grad=True)
    scale = auxiliary_output_gradient_scale(
        primary_gradient_norm=primary,
        auxiliary_gradient_norm=auxiliary,
        target_ratio=0.25,
        maximum_scale=10.0,
    )
    torch.testing.assert_close(scale, torch.tensor(0.5))
    torch.testing.assert_close(scale * auxiliary.detach() / primary.detach(), torch.tensor(0.25))
    assert not scale.requires_grad

    capped = auxiliary_output_gradient_scale(
        primary_gradient_norm=torch.tensor(100.0),
        auxiliary_gradient_norm=torch.tensor(0.1),
        target_ratio=0.5,
        maximum_scale=3.0,
    )
    torch.testing.assert_close(capped, torch.tensor(3.0))

    for primary_norm, auxiliary_norm, target_ratio in (
        (4.0, 2.0, 0.0),
        (0.0, 2.0, 0.25),
        (4.0, 0.0, 0.25),
    ):
        zero = auxiliary_output_gradient_scale(
            primary_gradient_norm=torch.tensor(primary_norm),
            auxiliary_gradient_norm=torch.tensor(auxiliary_norm),
            target_ratio=target_ratio,
            maximum_scale=10.0,
        )
        torch.testing.assert_close(zero, torch.tensor(0.0))


def test_loss_output_gradient_norm_measures_output_space_without_consuming_graph() -> None:
    outputs = torch.tensor([3.0, 4.0], requires_grad=True)
    loss = outputs.square().sum()
    output_gradient = loss_output_gradient(loss, outputs)
    output_gradient_norm = loss_output_gradient_norm(loss, outputs)

    # d/doutputs sum(outputs**2) = [6, 8], whose L2 norm is 10.
    torch.testing.assert_close(output_gradient, torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(output_gradient_norm, torch.tensor(10.0))
    assert not output_gradient.requires_grad
    assert not output_gradient_norm.requires_grad
    loss.backward()
    torch.testing.assert_close(outputs.grad, torch.tensor([6.0, 8.0]))


def test_current_equal_reference_has_zero_delta_kl_log_ratio_and_lambda_zero_loss() -> None:
    generator = torch.Generator().manual_seed(2)
    mean = torch.randn((3, 2, 3), generator=generator)
    action = torch.randn((3, 2, 3), generator=generator)
    variance = torch.tensor([0.12, 0.31, 0.67]).reshape(3, 1, 1)

    mean_delta = mean - mean
    log_ratio = transition_log_uplift(action, mean, mean, variance, reduction="sum")
    expected_log_ratio = expected_transition_log_uplift(
        mean, mean, mean, variance, reduction="sum"
    )
    kl = equal_covariance_gaussian_kl(mean, mean, variance, reduction="sum")
    lambda_zero_loss = scaled_basu_loss_from_log_ratio(
        log_ratio, bregman_lambda=0.0, bregman_scale=1.7
    )

    # A preference-pair delta is the difference of two per-sample log ratios.
    # It must vanish too when current and reference transitions coincide.
    preference_delta = log_ratio[0] - log_ratio[1]
    assert torch.count_nonzero(mean_delta) == 0
    torch.testing.assert_close(preference_delta, torch.zeros((), dtype=log_ratio.dtype))
    torch.testing.assert_close(log_ratio, torch.zeros_like(log_ratio))
    torch.testing.assert_close(expected_log_ratio, torch.zeros_like(expected_log_ratio))
    torch.testing.assert_close(kl, torch.zeros_like(kl))
    torch.testing.assert_close(lambda_zero_loss, torch.zeros_like(lambda_zero_loss))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_uplift_swap_symmetry_and_equal_covariance_kl_symmetry(reduction: str) -> None:
    generator = torch.Generator().manual_seed(3)
    action = torch.randn((2, 3, 4), generator=generator)
    current_mean = torch.randn((2, 3, 4), generator=generator)
    reference_mean = torch.randn((2, 3, 4), generator=generator)
    true_mean = torch.randn((2, 3, 4), generator=generator)
    variance = torch.tensor([0.17, 0.53]).reshape(2, 1, 1)

    uplift = transition_log_uplift(
        action, current_mean, reference_mean, variance, reduction=reduction
    )
    swapped_uplift = transition_log_uplift(
        action, reference_mean, current_mean, variance, reduction=reduction
    )
    expected_uplift = expected_transition_log_uplift(
        true_mean, current_mean, reference_mean, variance, reduction=reduction
    )
    swapped_expected_uplift = expected_transition_log_uplift(
        true_mean, reference_mean, current_mean, variance, reduction=reduction
    )

    torch.testing.assert_close(uplift, -swapped_uplift)
    torch.testing.assert_close(expected_uplift, -swapped_expected_uplift)
    torch.testing.assert_close(
        equal_covariance_gaussian_kl(reference_mean, current_mean, variance, reduction),
        equal_covariance_gaussian_kl(current_mean, reference_mean, variance, reduction),
    )


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_swapping_preferred_and_rejected_branches_negates_log_ratio(reduction: str) -> None:
    generator = torch.Generator().manual_seed(31)
    action = torch.randn((2, 2, 3), generator=generator)
    current_mean = torch.randn((2, 2, 3), generator=generator)
    reference_mean = torch.randn((2, 2, 3), generator=generator)
    variance = torch.tensor([0.23, 0.61]).reshape(2, 1, 1)
    beta = 2.7

    def pair_log_ratio(
        pair_action: torch.Tensor,
        pair_current_mean: torch.Tensor,
        pair_reference_mean: torch.Tensor,
        pair_variance: torch.Tensor,
    ) -> torch.Tensor:
        delta = transition_log_uplift(
            pair_action, pair_current_mean, pair_reference_mean, pair_variance, reduction
        )
        state_kl = equal_covariance_gaussian_kl(
            pair_reference_mean, pair_current_mean, pair_variance, reduction
        )
        return beta * (delta[1] - delta[0] + state_kl[1] - state_kl[0])

    log_ratio = pair_log_ratio(action, current_mean, reference_mean, variance)
    swapped_log_ratio = pair_log_ratio(
        action.flip(0), current_mean.flip(0), reference_mean.flip(0), variance.flip(0)
    )
    torch.testing.assert_close(swapped_log_ratio, -log_ratio)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_gaussian_uplift_and_kl_match_torch_distributions(reduction: str) -> None:
    generator = torch.Generator().manual_seed(4)
    action = torch.randn((2, 2, 3), generator=generator)
    current_mean = torch.randn((2, 2, 3), generator=generator)
    reference_mean = torch.randn((2, 2, 3), generator=generator)
    variance = torch.tensor([0.19, 0.71]).reshape(2, 1, 1)
    standard_deviation = variance.sqrt()

    reference_distribution = Normal(reference_mean, standard_deviation)
    current_distribution = Normal(current_mean, standard_deviation)
    distribution_uplift = (
        current_distribution.log_prob(action) - reference_distribution.log_prob(action)
    ).flatten(start_dim=1)
    distribution_kl = kl_divergence(reference_distribution, current_distribution).flatten(
        start_dim=1
    )
    if reduction == "sum":
        expected_uplift = distribution_uplift.sum(dim=1)
        expected_kl = distribution_kl.sum(dim=1)
    else:
        expected_uplift = distribution_uplift.mean(dim=1)
        expected_kl = distribution_kl.mean(dim=1)

    torch.testing.assert_close(
        transition_log_uplift(action, current_mean, reference_mean, variance, reduction),
        expected_uplift,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        equal_covariance_gaussian_kl(reference_mean, current_mean, variance, reduction),
        expected_kl,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_epsilon_space_kl_identity(reduction: str) -> None:
    alphas_cumprod = torch.tensor([0.99, 0.91, 0.79, 0.64, 0.48])
    timesteps = torch.tensor([1, 4])
    sample_shape = torch.Size((2, 2, 3, 4))
    coefficients = ddpm_posterior_coefficients(alphas_cumprod, timesteps, sample_shape)
    generator = torch.Generator().manual_seed(5)
    noisy_latents = torch.randn(sample_shape, generator=generator)
    epsilon_current = torch.randn(sample_shape, generator=generator)
    epsilon_reference = torch.randn(sample_shape, generator=generator)

    predicted_clean_current = predict_clean_from_epsilon(
        noisy_latents, epsilon_current, coefficients.alpha_t, coefficients.sigma_t
    )
    predicted_clean_reference = predict_clean_from_epsilon(
        noisy_latents, epsilon_reference, coefficients.alpha_t, coefficients.sigma_t
    )
    current_mean = transition_mean(noisy_latents, predicted_clean_current, coefficients)
    reference_mean = transition_mean(noisy_latents, predicted_clean_reference, coefficients)
    actual_kl = equal_covariance_gaussian_kl(
        reference_mean, current_mean, coefficients.variance, reduction
    )

    epsilon_gain = (
        coefficients.coef_x0 * coefficients.sigma_t / coefficients.alpha_t
    ).square() / coefficients.variance
    expected_kl = 0.5 * reduce_latent(
        (epsilon_current - epsilon_reference).square(), reduction
    ) * _per_example(epsilon_gain)
    torch.testing.assert_close(actual_kl, expected_kl, rtol=2e-5, atol=2e-5)


def test_corrected_x0_coefficient_uses_squared_signal_terms() -> None:
    # This deliberately coarse schedule makes the wrong, unsquared B formula
    # visibly different, preventing a quiet regression to the old expression.
    alphas_cumprod = torch.tensor([0.98, 0.75, 0.45, 0.12])
    timesteps = torch.tensor([3])
    coefficients = ddpm_posterior_coefficients(
        alphas_cumprod, timesteps, torch.Size((1, 2, 2))
    )

    alpha_s = alphas_cumprod[timesteps - 1].sqrt()
    alpha_t = alphas_cumprod[timesteps].sqrt()
    sigma_s = (1.0 - alphas_cumprod[timesteps - 1]).sqrt()
    sigma_t = (1.0 - alphas_cumprod[timesteps]).sqrt()
    corrected_b = (
        alpha_s.square() * sigma_t.square() - alpha_t.square() * sigma_s.square()
    ) / (alpha_s * sigma_t.square())
    incorrect_unsquared_b = (
        alpha_s * sigma_t.square() - alpha_t * sigma_s.square()
    ) / (alpha_s * sigma_t.square())

    actual_b = coefficients.coef_x0[:, 0, 0]
    torch.testing.assert_close(actual_b, corrected_b, rtol=2e-6, atol=2e-6)
    assert torch.max((corrected_b - incorrect_unsquared_b).abs()) > 0.05
    assert not torch.allclose(actual_b, incorrect_unsquared_b, rtol=1e-3, atol=1e-3)


def test_lambda_zero_scaled_basu_loss_matches_small_positive_lambda() -> None:
    log_ratio = torch.tensor([-1.2, -0.6, 0.4, 0.9])
    bregman_scale = 1.7
    lambda_zero = scaled_basu_loss_from_log_ratio(
        log_ratio, bregman_lambda=0.0, bregman_scale=bregman_scale, max_exp_argument=None
    )
    small_positive_lambda = scaled_basu_loss_from_log_ratio(
        log_ratio, bregman_lambda=1e-5, bregman_scale=bregman_scale, max_exp_argument=None
    )
    analytic_limit = (torch.expm1(log_ratio) + log_ratio) / bregman_scale

    torch.testing.assert_close(lambda_zero, analytic_limit)
    # The generic branch necessarily subtracts nearly equal float32 terms at
    # lambda=1e-5, so this is intentionally a numerical (not bitwise) limit.
    torch.testing.assert_close(lambda_zero, small_positive_lambda, rtol=1.1e-2, atol=2e-3)


@pytest.mark.parametrize("kappa", [0.5, 0.35, 0.2])
def test_mode_seeking_power_loss_matches_formula_and_scaled_basu_family(kappa: float) -> None:
    log_ratio = torch.tensor([-3.0, -1.2, 0.0, 0.7, 2.4])
    actual = mode_seeking_power_loss_from_log_ratio(
        log_ratio, kappa=kappa, max_exp_argument=None
    )
    expected = 0.5 * (
        torch.exp(kappa * log_ratio) / kappa
        + torch.exp((1.0 - kappa) * log_ratio) / (1.0 - kappa)
    )
    centered_scaled_basu = scaled_basu_loss_from_log_ratio(
        log_ratio,
        bregman_lambda=kappa - 1.0,
        bregman_scale=2.0,
        max_exp_argument=None,
    )
    additive_constant = 1.0 / (2.0 * kappa * (1.0 - kappa))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, centered_scaled_basu + additive_constant)


def test_hellinger_mode_specialization_and_margin_gradient_normalization() -> None:
    margin = torch.tensor([-2.0, 0.0, 1.5], requires_grad=True)
    loss = mode_seeking_power_loss_from_log_ratio(
        -margin, kappa=0.5, max_exp_argument=None
    )
    expected_loss = 2.0 * torch.exp(-margin / 2.0)
    torch.testing.assert_close(loss, expected_loss)

    loss.sum().backward()
    expected_margin_gradient = -torch.exp(-margin.detach() / 2.0)
    torch.testing.assert_close(margin.grad, expected_margin_gradient)
    torch.testing.assert_close(margin.grad[1], torch.tensor(-1.0))


def test_mode_seeking_kappa_symmetry_and_population_ratio_optimum() -> None:
    log_ratio = torch.linspace(-3.0, 3.0, 17)
    for kappa in (0.2, 0.35, 0.5):
        torch.testing.assert_close(
            mode_seeking_power_loss_from_log_ratio(
                log_ratio, kappa=kappa, max_exp_argument=None
            ),
            mode_seeking_power_loss_from_log_ratio(
                log_ratio, kappa=1.0 - kappa, max_exp_argument=None
            ),
        )

        preference_probability = 0.8
        optimum = math.log((1.0 - preference_probability) / preference_probability)
        candidate = torch.tensor(optimum, requires_grad=True)
        risk = preference_probability * mode_seeking_power_loss_from_log_ratio(
            candidate.reshape(1), kappa=kappa, max_exp_argument=None
        ) + (1.0 - preference_probability) * mode_seeking_power_loss_from_log_ratio(
            -candidate.reshape(1), kappa=kappa, max_exp_argument=None
        )
        gradient = torch.autograd.grad(risk.sum(), candidate)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-6, rtol=0)

        left = preference_probability * mode_seeking_power_loss_from_log_ratio(
            torch.tensor([optimum - 0.2]), kappa=kappa, max_exp_argument=None
        ) + (1.0 - preference_probability) * mode_seeking_power_loss_from_log_ratio(
            torch.tensor([-optimum + 0.2]), kappa=kappa, max_exp_argument=None
        )
        right = preference_probability * mode_seeking_power_loss_from_log_ratio(
            torch.tensor([optimum + 0.2]), kappa=kappa, max_exp_argument=None
        ) + (1.0 - preference_probability) * mode_seeking_power_loss_from_log_ratio(
            torch.tensor([-optimum - 0.2]), kappa=kappa, max_exp_argument=None
        )
        assert risk.item() < left.item()
        assert risk.item() < right.item()


def test_hellinger_mode_margin_gradient_dominates_logistic_ratio_gradient() -> None:
    ratio = torch.tensor([1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3])
    log_ratio = ratio.log()
    mode_seeking_gradient = mode_seeking_margin_gradient_magnitude(
        log_ratio, kappa=0.5, max_exp_argument=None
    )
    logistic_gradient = 2.0 * torch.sigmoid(log_ratio)

    torch.testing.assert_close(mode_seeking_gradient, ratio.sqrt())
    assert torch.all(mode_seeking_gradient >= logistic_gradient)
    torch.testing.assert_close(mode_seeking_gradient[3], logistic_gradient[3])
    assert torch.all(mode_seeking_gradient[:3] > logistic_gradient[:3])
    assert torch.all(mode_seeking_gradient[4:] > logistic_gradient[4:])


@pytest.mark.parametrize("kappa", [-1.0, 0.0, 1.0, 2.0, float("nan"), float("inf")])
def test_mode_seeking_kappa_must_be_strictly_inside_unit_interval(kappa: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        mode_seeking_power_loss_from_log_ratio(torch.zeros(1), kappa=kappa)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        mode_seeking_exponent_clip_fraction(torch.zeros(1), kappa, 30.0)


def test_mode_seeking_exponent_clipping_is_reported_and_finite() -> None:
    log_ratio = torch.tensor(
        [-100.0, -61.0, -60.0, -59.0, 0.0, 59.0, 60.0, 61.0, 100.0],
        requires_grad=True,
    )
    clipped = mode_seeking_exponent_clip_fraction(
        log_ratio, kappa=0.5, max_exp_argument=30.0
    )
    expected_clipped = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    loss = mode_seeking_power_loss_from_log_ratio(
        log_ratio, kappa=0.5, max_exp_argument=30.0
    )

    torch.testing.assert_close(clipped, expected_clipped)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert log_ratio.grad is not None
    assert torch.isfinite(log_ratio.grad).all()
    assert torch.all(log_ratio.grad > 0)
    torch.testing.assert_close(
        log_ratio.grad,
        mode_seeking_margin_gradient_magnitude(
            log_ratio.detach(), kappa=0.5, max_exp_argument=30.0
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_inputs_promote_to_finite_float32_with_gradients(dtype: torch.dtype) -> None:
    alphas_cumprod = torch.tensor([0.99, 0.94, 0.87, 0.79], dtype=dtype)
    timesteps = torch.tensor([1, 2])
    sample_shape = torch.Size((2, 2, 4))
    coefficients = ddpm_posterior_coefficients(alphas_cumprod, timesteps, sample_shape)
    generator = torch.Generator().manual_seed(6)
    # Generate in float32 first so the test is portable to CPU builds that do
    # not implement random sampling kernels for every reduced-precision type.
    noisy_latents = torch.randn(sample_shape, generator=generator).to(dtype)
    clean_latents = torch.randn(sample_shape, generator=generator).to(dtype)
    epsilon_current = torch.randn(sample_shape, generator=generator).to(dtype).requires_grad_()
    epsilon_reference = torch.randn(sample_shape, generator=generator).to(dtype)

    current_clean = predict_clean_from_epsilon(
        noisy_latents, epsilon_current, coefficients.alpha_t, coefficients.sigma_t
    )
    reference_clean = predict_clean_from_epsilon(
        noisy_latents, epsilon_reference, coefficients.alpha_t, coefficients.sigma_t
    )
    current_mean = transition_mean(noisy_latents, current_clean, coefficients)
    reference_mean = transition_mean(noisy_latents, reference_clean, coefficients)
    action = true_posterior_mean(noisy_latents, clean_latents, coefficients)
    log_ratio = transition_log_uplift(
        action, current_mean, reference_mean, coefficients.variance, reduction="mean"
    )
    kl = equal_covariance_gaussian_kl(
        reference_mean, current_mean, coefficients.variance, reduction="mean"
    )
    basu = scaled_basu_loss_from_log_ratio(log_ratio, bregman_lambda=0.2, bregman_scale=1.0)
    mode_seeking = mode_seeking_power_loss_from_log_ratio(log_ratio, kappa=0.35)

    for value in (
        coefficients.coef_xt,
        coefficients.coef_x0,
        coefficients.variance,
        current_clean,
        current_mean,
        log_ratio,
        kl,
        basu,
        mode_seeking,
    ):
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()

    (
        log_ratio.square().mean()
        + kl.mean()
        + basu.square().mean()
        + mode_seeking.square().mean()
    ).backward()
    assert epsilon_current.grad is not None
    assert torch.isfinite(epsilon_current.grad).all()


def test_timestep_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="timesteps.*>= 1"):
        ddpm_posterior_coefficients(
            torch.tensor([0.99, 0.91, 0.82]),
            torch.tensor([0]),
            torch.Size((1, 2, 2)),
        )


def test_original_diffusion_dpo_regression_on_fixed_synthetic_batch() -> None:
    """Exercise the runtime baseline helper on a fixed preferred-first batch."""
    model_losses = torch.tensor([0.75, 1.25, 1.10, 0.40])
    reference_losses = torch.tensor([0.60, 1.40, 0.90, 0.45])
    loss, logit = diffusion_dpo_loss_from_per_sample_losses(
        model_losses, reference_losses, beta_dpo=2.5
    )

    expected_logit = torch.tensor([0.0625, 0.1250])
    expected_loss = torch.tensor(
        (math.log1p(math.exp(-0.0625)) + math.log1p(math.exp(-0.1250))) / 2.0
    )
    torch.testing.assert_close(logit, expected_logit)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)

    # The training script must use this tested helper, rather than maintaining
    # a divergent DPO implementation beside ratio_bregman.
    train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")
    assert "dpo_loss, inside_term = diffusion_dpo_loss_from_per_sample_losses(" in train_source
    assert "loss = dpo_loss" in train_source
    assert 'if args.preference_loss == "dpo":' in train_source


def test_training_script_dispatches_explicit_mode_seeking_objective() -> None:
    train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")
    assert '"mode_seeking_ratio"' in train_source
    assert "mode_seeking_power_loss_from_log_ratio(" in train_source
    assert "kappa=args.mode_seeking_kappa" in train_source
    assert '"mode_seeking_kappa": (' in train_source
    assert "args.preference_loss in RATIO_PREFERENCE_LOSSES" in train_source


def test_training_script_dispatches_step_aware_tbpo_and_saves_its_head() -> None:
    train_source = (Path(__file__).resolve().parents[1] / "train.py").read_text(
        encoding="utf-8"
    )
    assert '"step_aware_tbpo"' in train_source
    assert "StepConfidenceHead(" in train_source
    assert "step_preference_consistency_target(" in train_source
    assert "step_aware_preference_loss(" in train_source
    assert '"lr": args.confidence_learning_rate' in train_source
    assert '"step_confidence_head.pt"' in train_source


def test_training_script_and_launcher_dispatch_mvp_controls() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_source = (project_root / "train.py").read_text(encoding="utf-8")
    for expected in (
        '"--ratio_margin_gradient_scale"',
        '"--confidence_policy_weight_normalization"',
        '"--reference_anchor_output_gradient_ratio"',
        '"--winner_anchor_output_gradient_ratio"',
        '"--pair_label_policy"',
        "preference_loss_margin_scale(",
        "symmetric_reference_kl_anchor(",
        "winner_denoising_anchor(",
        "auxiliary_output_gradient_scale(",
        "policy_weight_normalizer=policy_weight_normalizer",
    ):
        assert expected in train_source
    assert "with accelerator.accumulate(unet):" in train_source
    assert "accelerator.accumulate(*accumulation_models)" not in train_source

    launcher_source = (
        project_root / "hessian" / "run_sa_tbpo_mvp_85k_eff64.sh"
    ).read_text(encoding="utf-8")
    for expected in (
        "b3|t2|t3|t7|q2",
        "MAX_STEPS_DEFAULT=1329",
        "--gradient_accumulation_steps=16",
        "--pair_label_policy=filter",
        "--no_hflip",
        "--proportion_empty_prompts=0",
        "--confidence_policy_weight_normalization=global_microbatch_mean",
        'RATIO_BETA_DEFAULT=500',
        '--ratio_margin_gradient_scale="$RATIO_MARGIN_GRADIENT_SCALE_VALUE"',
    ):
        assert expected in launcher_source


def test_post_mvp_lpo_dspo_and_mode_seeking_branches_are_explicit() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_source = (project_root / "train.py").read_text(encoding="utf-8")
    for expected in (
        '"step_aware_mode_seeking_tbpo"',
        '"--winner_anchor_type"',
        '"--dspo_probability_source"',
        '"--dspo_logit_beta"',
        "dspo_winner_score_anchor(",
        '"--confidence_target_source"',
        '"--confidence_policy_signal"',
        "select_timestep_pair_scores(",
        "latent_teacher_preference_probability(",
        "args.preference_loss in STEP_AWARE_PREFERENCE_LOSSES",
        "args.preference_loss in MODE_SEEKING_PREFERENCE_LOSSES",
    ):
        assert expected in train_source

    launcher_source = (
        project_root / "hessian" / "run_lc_sa_tbpo_post_mvp_85k_eff64.sh"
    ).read_text(encoding="utf-8")
    for expected in (
        "q2_mvp|q3_dspo|l2_lrm|d1_power05|d2_power035",
        "--winner_anchor_type=dspo_score",
        "--confidence_target_source=precomputed_lrm",
        "--confidence_policy_signal=lrm_teacher",
        "--confidence_loss_weight=0",
        "validate_lrm_scores.py",
        "PREFERENCE_LOSS=step_aware_mode_seeking_tbpo",
        "--mode_seeking_kappa=0.5",
        "--mode_seeking_kappa=0.35",
        "--transition_estimator=pointwise",
        "--max_exp_argument=15",
    ):
        assert expected in launcher_source
