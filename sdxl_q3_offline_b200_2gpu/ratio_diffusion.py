"""Numerically stable math for transition-level Ratio Diffusion training.

The implementation deliberately uses the DDPM posterior
``q(z_{t-1} | z_t, z_0)`` as the fixed-variance reverse transition family.
All ratio calculations stay in log space and are promoted to float32 so the
module can be used from fp16/bf16 training loops.

The expanded posterior coefficients below are the corrected formulas for the
signal parameterization ``z_t = alpha_t z_0 + sigma_t epsilon``.  In
particular, the coefficient of ``z_0`` contains squared signal coefficients;
using the unsquared expression gives an inconsistent Gaussian posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Reduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class PosteriorCoefficients:
    """Broadcastable coefficients of ``q(z_{t-1} | z_t, z_0)``."""

    coef_xt: Tensor
    coef_x0: Tensor
    variance: Tensor
    alpha_t: Tensor
    sigma_t: Tensor


@dataclass(frozen=True)
class StepAwareLossOutput:
    """Components of the non-degenerate step-aware preference objective."""

    loss: Tensor
    policy_loss: Tensor
    confidence_loss: Tensor
    policy_weights: Tensor
    raw_policy_weights: Tensor
    policy_weight_normalizer: Tensor


def select_timestep_pair_scores(
    preferred_scores: Tensor,
    rejected_scores: Tensor,
    timesteps: Tensor,
    num_train_timesteps: int,
) -> Tuple[Tensor, Tensor]:
    """Select precomputed latent-teacher scores for the sampled timestep.

    Scores may be scalar per pair (shape ``[B]`` or ``[B, 1]``) or binned
    along diffusion time (shape ``[B, K]``).  For ``K`` bins, timestep ``t``
    is assigned to ``floor(t * K / T)``.  Consequently, ``K == T`` recovers
    exact per-timestep lookup while a small ``K`` supports inexpensive LRM
    calibration files.
    """
    if preferred_scores.shape != rejected_scores.shape:
        raise ValueError("preferred_scores and rejected_scores must have the same shape.")
    if preferred_scores.ndim not in (1, 2):
        raise ValueError("Pair scores must have shape [B] or [B, K].")
    if timesteps.ndim != 1 or timesteps.shape[0] != preferred_scores.shape[0]:
        raise ValueError("timesteps must have shape [B] matching the pair scores.")
    if preferred_scores.shape[0] == 0:
        raise ValueError("Pair scores must be non-empty.")
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be at least 2.")
    if torch.any(timesteps < 0) or torch.any(timesteps >= num_train_timesteps):
        raise ValueError(
            f"timesteps must lie in [0, {num_train_timesteps - 1}]."
        )

    preferred = preferred_scores.float()
    rejected = rejected_scores.float()
    if not bool(torch.isfinite(preferred).all()) or not bool(
        torch.isfinite(rejected).all()
    ):
        raise ValueError("Pair scores must contain only finite values.")
    if preferred.ndim == 1:
        return preferred, rejected

    num_bins = preferred.shape[1]
    if num_bins < 1:
        raise ValueError("Binned pair scores must contain at least one timestep bin.")
    if num_bins == 1:
        return preferred[:, 0], rejected[:, 0]
    bin_indices = torch.div(
        timesteps.long() * num_bins,
        num_train_timesteps,
        rounding_mode="floor",
    ).clamp(max=num_bins - 1)
    return (
        preferred.gather(1, bin_indices[:, None]).squeeze(1),
        rejected.gather(1, bin_indices[:, None]).squeeze(1),
    )


def latent_teacher_preference_probability(
    preferred_scores: Tensor,
    rejected_scores: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Convert frozen LRM scores into ``P(preferred > rejected)``.

    The probability is a detached teacher signal in the training loop.  This
    helper only performs the numerically stable Bradley--Terry conversion.
    """
    if preferred_scores.shape != rejected_scores.shape:
        raise ValueError("preferred_scores and rejected_scores must have the same shape.")
    if preferred_scores.ndim != 1:
        raise ValueError("Selected latent-teacher scores must have shape [B].")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    preferred = preferred_scores.float()
    rejected = rejected_scores.float()
    if not bool(torch.isfinite(preferred).all()) or not bool(
        torch.isfinite(rejected).all()
    ):
        raise ValueError("Selected latent-teacher scores must be finite.")
    return torch.sigmoid((preferred - rejected) / temperature)


def sinusoidal_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    num_train_timesteps: int,
    max_period: float = 10_000.0,
) -> Tensor:
    """Return a scheduler-length-independent sinusoidal timestep embedding."""
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}.")
    if embedding_dim < 2 or embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be an even integer >= 2.")
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be at least 2.")
    if max_period <= 0 or not math.isfinite(max_period):
        raise ValueError("max_period must be positive and finite.")
    if torch.any(timesteps < 0) or torch.any(timesteps >= num_train_timesteps):
        raise ValueError(
            f"timesteps must lie in [0, {num_train_timesteps - 1}]."
        )

    # Normalize different scheduler lengths onto the conventional [0, 1000]
    # diffusion-time interval before applying the Fourier features.
    normalized_time = (
        timesteps.float() * (1000.0 / float(num_train_timesteps - 1))
    )
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / float(half_dim)
    )
    angles = normalized_time[:, None] * frequencies[None, :]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


class StepConfidenceHead(nn.Module):
    """Small pairwise head estimating local preference reliability.

    ``features`` are preferred-first/rejected-second spatial feature maps. In
    training they are built from the noisy latent and frozen reference UNet output,
    so the head reuses diffusion representations without adding hooks to the
    backbone. Spatial means and standard deviations keep the parameter and
    activation overhead independent of image resolution.
    """

    def __init__(
        self,
        feature_channels: int,
        timestep_embedding_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive.")
        if timestep_embedding_dim < 2 or timestep_embedding_dim % 2 != 0:
            raise ValueError("timestep_embedding_dim must be an even integer >= 2.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")

        self.feature_channels = int(feature_channels)
        self.timestep_embedding_dim = int(timestep_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        # Each branch contributes mean and standard deviation (2C). Keep both
        # ordered branches, their signed difference, and absolute difference.
        pair_feature_dim = 8 * self.feature_channels
        input_dim = pair_feature_dim + self.timestep_embedding_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Start from an uninformative confidence of exactly 0.5. This avoids
        # injecting a random timestep bias before the BCE head has learned.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _spatial_moments(features: Tensor) -> Tensor:
        if features.ndim < 3:
            raise ValueError("features must have shape [B, C, ...].")
        flattened = features.float().flatten(start_dim=2)
        return torch.cat(
            [
                flattened.mean(dim=2),
                flattened.std(dim=2, unbiased=False),
            ],
            dim=1,
        )

    def forward(
        self,
        features: Tensor,
        pair_timesteps: Tensor,
        num_train_timesteps: int,
    ) -> Tensor:
        if features.ndim < 3:
            raise ValueError("features must have shape [2B, C, ...].")
        if features.shape[0] == 0 or features.shape[0] % 2 != 0:
            raise ValueError(
                "features must be a non-empty preferred-first/rejected-second batch."
            )
        if features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, got {features.shape[1]}."
            )
        pair_batch_size = features.shape[0] // 2
        if pair_timesteps.shape != (pair_batch_size,):
            raise ValueError(
                f"pair_timesteps must have shape [{pair_batch_size}], "
                f"got {tuple(pair_timesteps.shape)}."
            )

        preferred, rejected = self._spatial_moments(features).chunk(2)
        difference = preferred - rejected
        pair_features = torch.cat(
            [preferred, rejected, difference, difference.abs()], dim=1
        )
        time_features = sinusoidal_timestep_embedding(
            pair_timesteps,
            embedding_dim=self.timestep_embedding_dim,
            num_train_timesteps=num_train_timesteps,
        )
        logits = self.network(torch.cat([pair_features, time_features], dim=1))
        return torch.sigmoid(logits.squeeze(1).float()).clamp(1e-6, 1.0 - 1e-6)


def step_preference_consistency_target(
    reference_losses: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Build ``z_t`` from reference denoising consistency for each pair.

    A local step is consistent when the frozen reference transition predicts
    the preferred branch with lower denoising error than the rejected branch.
    Ties remain maximally uncertain (0.5). Optional symmetric label smoothing
    reduces sensitivity to noisy pair labels.
    """
    if reference_losses.ndim != 1:
        raise ValueError("reference_losses must be a one-dimensional tensor.")
    if reference_losses.shape[0] == 0 or reference_losses.shape[0] % 2 != 0:
        raise ValueError(
            "reference_losses must be a non-empty preferred-first/rejected-second batch."
        )
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1).")
    if not bool(torch.isfinite(reference_losses).all()):
        raise ValueError("reference_losses must contain only finite values.")

    preferred, rejected = reference_losses.float().chunk(2)
    target = torch.where(
        rejected > preferred,
        torch.ones_like(preferred),
        torch.where(
            rejected < preferred,
            torch.zeros_like(preferred),
            torch.full_like(preferred, 0.5),
        ),
    )
    return target * (1.0 - label_smoothing) + 0.5 * label_smoothing


def step_aware_preference_loss(
    per_pair_loss: Tensor,
    confidence: Tensor,
    confidence_target: Tensor,
    confidence_loss_weight: float = 1.0,
    minimum_policy_weight: float = 0.05,
    policy_weight_normalizer: Optional[Tensor] = None,
    policy_confidence: Optional[Tensor] = None,
) -> StepAwareLossOutput:
    """Combine weighted TBPO and confidence BCE without weight collapse.

    The confidence used in the policy term is detached deliberately. If its
    gradient were left attached, the head could reduce the objective simply by
    driving every weight toward zero rather than learning step reliability.
    The positive floor implements the ``w_t > 0`` assumption and prevents a
    temporarily uncertain head from deleting all policy gradients.
    """
    if per_pair_loss.ndim != 1:
        raise ValueError("per_pair_loss must be one-dimensional.")
    if confidence.shape != per_pair_loss.shape:
        raise ValueError("confidence must have the same shape as per_pair_loss.")
    if confidence_target.shape != per_pair_loss.shape:
        raise ValueError("confidence_target must have the same shape as per_pair_loss.")
    if not math.isfinite(confidence_loss_weight) or confidence_loss_weight < 0:
        raise ValueError("confidence_loss_weight must be non-negative and finite.")
    if (
        not math.isfinite(minimum_policy_weight)
        or not 0.0 < minimum_policy_weight < 1.0
    ):
        raise ValueError("minimum_policy_weight must lie strictly between 0 and 1.")
    if not bool(torch.isfinite(per_pair_loss).all()):
        raise ValueError("per_pair_loss must contain only finite values.")
    if not bool(torch.isfinite(confidence).all()) or bool(
        ((confidence <= 0) | (confidence >= 1)).any()
    ):
        raise ValueError("confidence must contain finite probabilities strictly inside (0, 1).")
    if not bool(torch.isfinite(confidence_target).all()) or bool(
        ((confidence_target < 0) | (confidence_target > 1)).any()
    ):
        raise ValueError("confidence_target must contain finite values in [0, 1].")

    if policy_confidence is None:
        policy_signal = confidence.float()
    else:
        if policy_confidence.shape != per_pair_loss.shape:
            raise ValueError("policy_confidence must have the same shape as per_pair_loss.")
        if not bool(torch.isfinite(policy_confidence).all()) or bool(
            ((policy_confidence < 0) | (policy_confidence > 1)).any()
        ):
            raise ValueError("policy_confidence must contain finite values in [0, 1].")
        policy_signal = policy_confidence.float()

    raw_policy_weights = minimum_policy_weight + (
        1.0 - minimum_policy_weight
    ) * policy_signal
    if policy_weight_normalizer is None:
        normalizer = torch.ones(
            (), device=raw_policy_weights.device, dtype=torch.float32
        )
    else:
        normalizer = torch.as_tensor(
            policy_weight_normalizer,
            device=raw_policy_weights.device,
            dtype=torch.float32,
        )
        if normalizer.numel() != 1:
            raise ValueError("policy_weight_normalizer must be a scalar.")
        normalizer = normalizer.reshape(())
        if not bool(torch.isfinite(normalizer)) or bool(normalizer <= 0):
            raise ValueError(
                "policy_weight_normalizer must be positive and finite."
            )
        normalizer = normalizer.detach()
    policy_weights = raw_policy_weights / normalizer
    policy_loss = (policy_weights.detach() * per_pair_loss.float()).mean()
    confidence_loss = F.binary_cross_entropy(
        confidence.float(), confidence_target.float()
    )
    loss = policy_loss + float(confidence_loss_weight) * confidence_loss
    return StepAwareLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        confidence_loss=confidence_loss,
        policy_weights=policy_weights,
        raw_policy_weights=raw_policy_weights,
        policy_weight_normalizer=normalizer,
    )


def preference_loss_margin_scale(
    ratio_beta: float,
    loss_log_ratio_slope_at_zero: float,
    target_margin_gradient: float,
) -> float:
    """Return a constant that decouples ratio temperature from update scale.

    If ``z = ratio_beta * margin`` and the unscaled preference loss has
    derivative ``loss_log_ratio_slope_at_zero`` with respect to ``z`` at the
    origin, multiplying the loss by the returned value makes its derivative
    with respect to the raw margin equal ``target_margin_gradient`` at zero.
    The multiplier is a positive constant, so it does not change the minimizer
    for a fixed ratio temperature.
    """
    values = {
        "ratio_beta": float(ratio_beta),
        "loss_log_ratio_slope_at_zero": float(loss_log_ratio_slope_at_zero),
        "target_margin_gradient": float(target_margin_gradient),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    return values["target_margin_gradient"] / (
        values["ratio_beta"] * values["loss_log_ratio_slope_at_zero"]
    )


def symmetric_reference_kl_anchor(per_sample_kl: Tensor) -> Tensor:
    """Return ``mean_pairs(KL_w + KL_l)`` for a preferred-first packed batch."""
    if per_sample_kl.ndim != 1:
        raise ValueError("per_sample_kl must be one-dimensional.")
    if per_sample_kl.shape[0] == 0 or per_sample_kl.shape[0] % 2 != 0:
        raise ValueError(
            "per_sample_kl must be a non-empty preferred-first/rejected-second batch."
        )
    values = per_sample_kl.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("per_sample_kl must contain only finite values.")
    if bool((values < -1e-6).any()):
        raise ValueError("per_sample_kl must be non-negative up to numerical tolerance.")
    preferred, rejected = values.chunk(2)
    return (preferred + rejected).mean()


def winner_denoising_anchor(model_prediction: Tensor, target: Tensor) -> Tensor:
    """Return float32 denoising MSE on only the preferred half of a packed pair batch."""
    if model_prediction.shape != target.shape:
        raise ValueError("model_prediction and target must have the same shape.")
    if model_prediction.ndim < 2:
        raise ValueError("model_prediction and target must include batch and feature dimensions.")
    if model_prediction.shape[0] == 0 or model_prediction.shape[0] % 2 != 0:
        raise ValueError(
            "model_prediction must be a non-empty preferred-first/rejected-second batch."
        )
    prediction = model_prediction.float()
    target_float = target.float()
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target_float).all()
    ):
        raise ValueError("model_prediction and target must contain only finite values.")
    pair_batch_size = prediction.shape[0] // 2
    return (prediction[:pair_batch_size] - target_float[:pair_batch_size]).square().mean()


def dspo_winner_score_anchor(
    model_prediction: Tensor,
    reference_prediction: Tensor,
    target: Tensor,
    preferred_probability: Tensor,
    correction_scale: float = 1.0,
) -> Tensor:
    """Return the DSPO-shaped score anchor on the preferred branch.

    For each pair this implements

    ``||(eps_theta^w - eps) - mu * (1 - p_w) *``
    ``(eps_theta^w - eps_ref^w)||^2``.

    ``preferred_probability`` is deliberately detached inside the helper.  It
    may be obtained from the TBPO transition margin or a frozen latent reward
    model, but this auxiliary must not become a second preference objective.
    At reference initialization the correction is zero, so the anchor reduces
    exactly to preferred-image denoising MSE.
    """
    if model_prediction.shape != reference_prediction.shape or model_prediction.shape != target.shape:
        raise ValueError(
            "model_prediction, reference_prediction, and target must have the same shape."
        )
    if model_prediction.ndim < 2:
        raise ValueError("Predictions and target must include batch and feature dimensions.")
    if model_prediction.shape[0] == 0 or model_prediction.shape[0] % 2 != 0:
        raise ValueError(
            "model_prediction must be a non-empty preferred-first/rejected-second batch."
        )
    pair_batch_size = model_prediction.shape[0] // 2
    if preferred_probability.shape != (pair_batch_size,):
        raise ValueError(
            f"preferred_probability must have shape [{pair_batch_size}]."
        )
    correction_scale = float(correction_scale)
    if not math.isfinite(correction_scale) or not 0.0 <= correction_scale < 1.0:
        raise ValueError("correction_scale must lie in [0, 1).")

    prediction = model_prediction.float()
    reference = reference_prediction.float()
    target_float = target.float()
    probability = preferred_probability.detach().float()
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(reference).all()
    ) or not bool(torch.isfinite(target_float).all()):
        raise ValueError("Predictions and target must contain only finite values.")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("preferred_probability must contain finite values in [0, 1].")

    preferred_prediction = prediction[:pair_batch_size]
    preferred_reference = reference[:pair_batch_size]
    preferred_target = target_float[:pair_batch_size]
    broadcast_shape = (pair_batch_size,) + (1,) * (prediction.ndim - 1)
    actionability = (1.0 - probability).reshape(broadcast_shape)
    score_residual = (
        preferred_prediction
        - preferred_target
        - correction_scale
        * actionability
        * (preferred_prediction - preferred_reference)
    )
    return score_residual.square().mean()


def loss_output_gradient(loss: Tensor, outputs: Tensor) -> Tensor:
    """Return the detached float32 tensor ``d loss / d outputs``.

    This inexpensive output-space proxy is used to allocate an explicit
    gradient budget to auxiliary losses without a second UNet backward pass.
    """
    if loss.numel() != 1:
        raise ValueError("loss must be scalar.")
    if not outputs.requires_grad:
        raise ValueError("outputs must require gradients.")
    gradient = torch.autograd.grad(
        loss,
        outputs,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradient.float().detach()


def loss_output_gradient_norm(loss: Tensor, outputs: Tensor) -> Tensor:
    """Return a detached float32 L2 norm of ``d loss / d outputs``."""
    return torch.linalg.vector_norm(loss_output_gradient(loss, outputs))


def auxiliary_output_gradient_scale(
    primary_gradient_norm: Tensor,
    auxiliary_gradient_norm: Tensor,
    target_ratio: float,
    maximum_scale: float,
    epsilon: float = 1e-12,
) -> Tensor:
    """Scale an auxiliary to a target output-gradient ratio, with a safety cap."""
    target_ratio = float(target_ratio)
    maximum_scale = float(maximum_scale)
    epsilon = float(epsilon)
    if not math.isfinite(target_ratio) or target_ratio < 0:
        raise ValueError("target_ratio must be non-negative and finite.")
    if not math.isfinite(maximum_scale) or maximum_scale <= 0:
        raise ValueError("maximum_scale must be positive and finite.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    primary = torch.as_tensor(primary_gradient_norm, dtype=torch.float32)
    auxiliary = torch.as_tensor(
        auxiliary_gradient_norm, device=primary.device, dtype=torch.float32
    )
    if primary.numel() != 1 or auxiliary.numel() != 1:
        raise ValueError("gradient norms must be scalar.")
    primary = primary.reshape(())
    auxiliary = auxiliary.reshape(())
    if not bool(torch.isfinite(primary)) or not bool(torch.isfinite(auxiliary)):
        raise ValueError("gradient norms must be finite.")
    if bool(primary < 0) or bool(auxiliary < 0):
        raise ValueError("gradient norms must be non-negative.")
    if target_ratio == 0 or bool(primary == 0) or bool(auxiliary <= epsilon):
        return torch.zeros_like(primary)
    return (target_ratio * primary / auxiliary.clamp_min(epsilon)).clamp(
        max=maximum_scale
    ).detach()


def extract_schedule_value(values: Tensor, timesteps: Tensor, sample_shape: torch.Size) -> Tensor:
    """Select one scalar schedule value per batch item and broadcast it.

    Args:
        values: One-dimensional scheduler values indexed by diffusion timestep.
        timesteps: Integer tensor of shape ``[batch]``.
        sample_shape: Shape of the latent batch that will receive the result.
    """
    if values.ndim != 1:
        raise ValueError(f"Scheduler values must be one-dimensional, got {tuple(values.shape)}.")
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}.")
    if len(sample_shape) < 1:
        raise ValueError("sample_shape must include a batch dimension.")
    if sample_shape[0] != timesteps.shape[0]:
        raise ValueError(
            "sample_shape batch dimension must match timesteps: "
            f"{sample_shape[0]} != {timesteps.shape[0]}."
        )
    if not torch.is_floating_point(values):
        values = values.float()
    if torch.any(timesteps < 0) or torch.any(timesteps >= values.shape[0]):
        raise ValueError(
            f"Timesteps must lie in [0, {values.shape[0] - 1}], got "
            f"min={int(timesteps.min())}, max={int(timesteps.max())}."
        )

    selected = values.to(device=timesteps.device, dtype=torch.float32)[timesteps.long()]
    return selected.reshape(timesteps.shape[0], *([1] * (len(sample_shape) - 1)))


def ddpm_posterior_coefficients(
    alphas_cumprod: Tensor,
    timesteps: Tensor,
    sample_shape: torch.Size,
    variance_floor: float = 1e-12,
) -> PosteriorCoefficients:
    """Return coefficients for the adjacent DDPM posterior.

    The returned parameters satisfy

    ``q(z_{t-1}|z_t,z_0) = Normal(A z_t + B z_0, v I)``.

    ``t == 0`` is intentionally rejected: it has no non-degenerate previous
    reverse transition in this estimator.
    """
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive.")
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}.")
    if torch.any(timesteps < 1):
        raise ValueError("Ratio transition training requires all timesteps to be >= 1.")

    previous_timesteps = timesteps - 1
    alpha_bar_t = extract_schedule_value(alphas_cumprod, timesteps, sample_shape)
    alpha_bar_s = extract_schedule_value(alphas_cumprod, previous_timesteps, sample_shape)

    one_minus_alpha_bar_t = (1.0 - alpha_bar_t).clamp_min(variance_floor)
    one_minus_alpha_bar_s = (1.0 - alpha_bar_s).clamp_min(0.0)
    transition_alpha_bar = alpha_bar_t / alpha_bar_s.clamp_min(variance_floor)
    transition_variance = (1.0 - transition_alpha_bar).clamp_min(0.0)

    # Standard DDPM posterior coefficients.  This is equivalent to
    # A = alpha_t sigma_s^2 / (alpha_s sigma_t^2),
    # B = (alpha_s^2 sigma_t^2 - alpha_t^2 sigma_s^2) /
    #     (alpha_s sigma_t^2), and
    # v = sigma_s^2 (1 - alpha_t^2 sigma_s^2 /
    #                    (alpha_s^2 sigma_t^2)).
    coef_xt = transition_alpha_bar.sqrt() * one_minus_alpha_bar_s / one_minus_alpha_bar_t
    coef_x0 = alpha_bar_s.sqrt() * transition_variance / one_minus_alpha_bar_t
    posterior_variance = (
        one_minus_alpha_bar_s * transition_variance / one_minus_alpha_bar_t
    ).clamp_min(variance_floor)

    return PosteriorCoefficients(
        coef_xt=coef_xt,
        coef_x0=coef_x0,
        variance=posterior_variance,
        alpha_t=alpha_bar_t.sqrt(),
        sigma_t=one_minus_alpha_bar_t.sqrt(),
    )


def predict_clean_from_epsilon(
    noisy_latents: Tensor,
    epsilon_prediction: Tensor,
    alpha_t: Tensor,
    sigma_t: Tensor,
) -> Tensor:
    """Convert an epsilon prediction to ``z_0`` under a VP scheduler."""
    return (noisy_latents.float() - sigma_t.float() * epsilon_prediction.float()) / alpha_t.float().clamp_min(1e-12)


def transition_mean(
    noisy_latents: Tensor,
    predicted_clean_latents: Tensor,
    coefficients: PosteriorCoefficients,
) -> Tensor:
    """Mean of the fixed-variance model transition conditioned on ``z_t``."""
    return (
        coefficients.coef_xt.float() * noisy_latents.float()
        + coefficients.coef_x0.float() * predicted_clean_latents.float()
    )


def true_posterior_mean(
    noisy_latents: Tensor,
    clean_latents: Tensor,
    coefficients: PosteriorCoefficients,
) -> Tensor:
    """Mean of the forward posterior anchored at the observed clean latent."""
    return (
        coefficients.coef_xt.float() * noisy_latents.float()
        + coefficients.coef_x0.float() * clean_latents.float()
    )


def sample_true_posterior_action(
    noisy_latents: Tensor,
    clean_latents: Tensor,
    posterior_noise: Tensor,
    coefficients: PosteriorCoefficients,
) -> Tuple[Tensor, Tensor]:
    """Sample ``z_s=A z_t+B z_0+sqrt(v) eta`` and return its true mean.

    ``A``, ``B``, and ``v`` come from :func:`ddpm_posterior_coefficients`, so
    this uses the corrected squared-signal posterior rather than the
    unsquared sampling expression in the early draft.
    """
    mean = true_posterior_mean(noisy_latents, clean_latents, coefficients)
    action = mean + coefficients.variance.float().sqrt() * posterior_noise.float()
    return action, mean


def reduce_latent(value: Tensor, reduction: Reduction) -> Tensor:
    """Reduce all non-batch latent dimensions without losing float32 precision."""
    flat = value.float().flatten(start_dim=1)
    if reduction == "sum":
        return flat.sum(dim=1)
    if reduction == "mean":
        return flat.mean(dim=1)
    raise ValueError(f"Unknown ratio reduction: {reduction!r}.")


def _per_example_variance(variance: Tensor) -> Tensor:
    if variance.ndim < 1:
        raise ValueError("variance must include a batch dimension.")
    return variance.float().flatten(start_dim=1)[:, 0]


def transition_log_uplift(
    action: Tensor,
    current_mean: Tensor,
    reference_mean: Tensor,
    variance: Tensor,
    reduction: Reduction,
) -> Tensor:
    """Return ``log p_theta(action|state) - log p_ref(action|state)``.

    Both policies use the same fixed covariance, so Gaussian normalization
    constants cancel exactly.  Direct squared-error differences are more
    stable than materializing high-dimensional log densities.
    """
    current_error = reduce_latent((action.float() - current_mean.float()).square(), reduction)
    reference_error = reduce_latent((action.float() - reference_mean.float()).square(), reduction)
    return -0.5 * (current_error - reference_error) / _per_example_variance(variance)


def expected_transition_log_uplift(
    true_mean: Tensor,
    current_mean: Tensor,
    reference_mean: Tensor,
    variance: Tensor,
    reduction: Reduction,
) -> Tensor:
    """Return the expected transition uplift under the true forward posterior.

    This is the lower-variance expected-log-ratio surrogate, not the exact
    pointwise objective because the Bregman loss is nonlinear.
    """
    current_error = reduce_latent((true_mean.float() - current_mean.float()).square(), reduction)
    reference_error = reduce_latent((true_mean.float() - reference_mean.float()).square(), reduction)
    return -0.5 * (current_error - reference_error) / _per_example_variance(variance)


def equal_covariance_gaussian_kl(
    reference_mean: Tensor,
    current_mean: Tensor,
    variance: Tensor,
    reduction: Reduction,
) -> Tensor:
    """Return ``KL(p_ref || p_theta)`` for equal-covariance Gaussians."""
    squared_distance = reduce_latent((reference_mean.float() - current_mean.float()).square(), reduction)
    return 0.5 * squared_distance / _per_example_variance(variance)


def diffusion_dpo_loss_from_per_sample_losses(
    model_losses: Tensor,
    reference_losses: Tensor,
    beta_dpo: float,
) -> Tuple[Tensor, Tensor]:
    """Return ``(loss, logit)`` for original Diffusion-DPO on a packed batch.

    This compact baseline helper is kept alongside the ratio math so the
    training path can have a numerical regression test without coupling that
    test to the full diffusers/accelerate stack.  It is intentionally the same
    MSE-difference expression used by the original training script.
    """
    if model_losses.ndim != 1 or reference_losses.ndim != 1:
        raise ValueError("Diffusion-DPO losses must be one-dimensional per-example tensors.")
    if model_losses.shape != reference_losses.shape:
        raise ValueError("Model and reference losses must have the same shape.")
    if model_losses.shape[0] == 0 or model_losses.shape[0] % 2 != 0:
        raise ValueError("Diffusion-DPO requires a non-empty preferred-first/rejected-second batch.")

    model_losses_w, model_losses_l = model_losses.chunk(2)
    reference_losses_w, reference_losses_l = reference_losses.chunk(2)
    model_diff = model_losses_w - model_losses_l
    reference_diff = reference_losses_w - reference_losses_l
    logit = -0.5 * float(beta_dpo) * (model_diff - reference_diff)
    return -torch.nn.functional.logsigmoid(logit).mean(), logit


def _validate_mode_seeking_inputs(
    log_ratio: Tensor,
    kappa: float,
    max_exp_argument: Optional[float],
) -> Tuple[Tensor, float]:
    kappa = float(kappa)
    if not math.isfinite(kappa) or not 0.0 < kappa < 1.0:
        raise ValueError("kappa must lie strictly between 0 and 1.")
    if max_exp_argument is not None and (
        not math.isfinite(max_exp_argument) or max_exp_argument <= 0
    ):
        raise ValueError("max_exp_argument must be positive and finite, or None.")
    values = log_ratio.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("log_ratio must contain only finite values.")
    return values, kappa


def mode_seeking_power_loss_from_log_ratio(
    log_ratio: Tensor,
    kappa: float = 0.5,
    max_exp_argument: Optional[float] = 30.0,
) -> Tensor:
    """Return the theorem-preserving mode-seeking power loss per pair.

    For ``R = exp(log_ratio)`` and ``0 < kappa < 1``, this implements

    ``0.5 * (R**kappa / kappa + R**(1-kappa) / (1-kappa))``.

    The recommended ``kappa=0.5`` specialization is ``2 * sqrt(R)``.  The
    calculation stays in log space and is promoted to float32.  A
    straight-through upper cap prevents overflow without turning confidently
    wrong pairs into zero-gradient examples.  The cap is a numerical guard,
    not part of the mathematical objective.
    """
    log_ratio, kappa = _validate_mode_seeking_inputs(
        log_ratio, kappa, max_exp_argument
    )

    def safe_exp(exponent: Tensor) -> Tensor:
        if max_exp_argument is not None:
            capped = exponent.clamp(max=max_exp_argument)
            # Forward uses the finite cap while backward follows the identity
            # into exp(capped), retaining a finite gradient in the corrective
            # direction for large positive log-ratios.
            exponent = capped.detach() + (exponent - exponent.detach())
        return torch.exp(exponent)

    return 0.5 * (
        safe_exp(kappa * log_ratio) / kappa
        + safe_exp((1.0 - kappa) * log_ratio) / (1.0 - kappa)
    )


def mode_seeking_exponent_clip_fraction(
    log_ratio: Tensor,
    kappa: float,
    max_exp_argument: Optional[float],
) -> Tensor:
    """Per-pair indicator that a mode-seeking loss exponent was clipped."""
    values, kappa = _validate_mode_seeking_inputs(
        log_ratio, kappa, max_exp_argument
    )
    if max_exp_argument is None:
        return torch.zeros_like(values)
    first = kappa * values > max_exp_argument
    second = (1.0 - kappa) * values > max_exp_argument
    return (first | second).float()


def mode_seeking_margin_gradient_magnitude(
    log_ratio: Tensor,
    kappa: float,
    max_exp_argument: Optional[float] = 30.0,
) -> Tensor:
    """Return the analytic ``|d loss / d v|`` for ``v = -log_ratio``.

    The same straight-through upper cap as the loss is used, so this is the
    gradient delivered to the optimizer even when the capped forward value is
    no longer the exact Eq. 27 objective.  Consult
    :func:`mode_seeking_exponent_clip_fraction` to detect frequent intervention
    by that numerical guard.
    """
    values, kappa = _validate_mode_seeking_inputs(
        log_ratio, kappa, max_exp_argument
    )
    first = kappa * values
    second = (1.0 - kappa) * values
    if max_exp_argument is not None:
        first = first.clamp(max=max_exp_argument)
        second = second.clamp(max=max_exp_argument)
    return 0.5 * (torch.exp(first) + torch.exp(second))


def scaled_basu_loss_from_log_ratio(
    log_ratio: Tensor,
    bregman_lambda: float,
    bregman_scale: float,
    max_exp_argument: Optional[float] = 30.0,
) -> Tensor:
    """Compute one Scaled-Basu Bregman loss value per preference pair.

    ``max_exp_argument`` only bounds exponential arguments; the linear
    ``log_ratio`` term is kept untouched.  That clamp is a numerical guard,
    not part of the theoretical objective.
    """
    if bregman_scale <= 0:
        raise ValueError("bregman_scale must be positive.")
    if max_exp_argument is not None and max_exp_argument <= 0:
        raise ValueError("max_exp_argument must be positive or None.")
    if abs(bregman_lambda + 1.0) < 1e-6:
        raise NotImplementedError("The bregman_lambda -> -1 limit is not implemented.")

    log_ratio = log_ratio.float()

    def safe_exp(exponent: Tensor) -> Tensor:
        if max_exp_argument is not None:
            exponent = exponent.clamp(-max_exp_argument, max_exp_argument)
        return torch.exp(exponent)

    def safe_expm1(exponent: Tensor) -> Tensor:
        if max_exp_argument is not None:
            exponent = exponent.clamp(-max_exp_argument, max_exp_argument)
        return torch.expm1(exponent)

    if abs(bregman_lambda) < 1e-6:
        # lim_{lambda -> 0} ell_SBA(R) = (R - 1 + log R) / c_B.
        return (safe_expm1(log_ratio) + log_ratio) / bregman_scale

    lam = float(bregman_lambda)
    numerator = (
        lam * safe_exp((1.0 + lam) * log_ratio)
        - (1.0 + lam) * safe_exp(-lam * log_ratio)
        + 1.0
    )
    return numerator / (bregman_scale * lam * (lam + 1.0))


def scaled_basu_exponent_clip_fraction(
    log_ratio: Tensor,
    bregman_lambda: float,
    max_exp_argument: Optional[float],
) -> Tensor:
    """Per-pair indicator that an SBA exponential argument was clipped."""
    values = log_ratio.float()
    if max_exp_argument is None:
        return torch.zeros_like(values)
    if max_exp_argument <= 0:
        raise ValueError("max_exp_argument must be positive or None.")
    if abs(bregman_lambda) < 1e-6:
        return (values.abs() > max_exp_argument).float()
    first = ((1.0 + bregman_lambda) * values).abs() > max_exp_argument
    second = (-bregman_lambda * values).abs() > max_exp_argument
    return (first | second).float()
