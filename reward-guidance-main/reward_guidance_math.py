"""Shared mathematics for positive-reward exponential tilting.

The target distribution used throughout the project is

    p_beta(x | c) = p_data(x | c) exp(beta r(x, c)) / Z_beta(c).

For a Gaussian component N(mu, variance I), a local Taylor expansion of the
reward gives the corrected (positive-reward) formulas

    first order:  mean = mu + beta * variance * grad_r
    second order: covariance = (variance^-1 I - beta H_r)^-1
                  mean = mu + beta * covariance @ grad_r.

The signs are intentionally the opposite of a cost-minimisation tilt
``exp(-beta r)``.  This module is dependency-light so both the toy experiments
and the H200 FLUX pipeline use the same implementation and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionDiagnostics:
    """Numerical diagnostics for a stabilized local Gaussian approximation."""

    min_eigenvalue_before: float
    min_eigenvalue_after: float
    num_clipped_curvatures: int


@dataclass(frozen=True)
class LowRankPrecisionDiagnostics:
    """Diagnostics for a low-rank feature-space precision correction."""

    min_eigenvalue_before: float
    min_eigenvalue_after: float
    positive_curvature_scale: float


def exponential_tilt_probabilities(rewards: torch.Tensor, beta: float) -> torch.Tensor:
    """Return normalized probabilities proportional to ``exp(beta * reward)``.

    The max subtraction is the finite-sample equivalent of evaluating the
    partition function in log space and prevents overflow for large beta.
    """

    if beta < 0.0:
        raise ValueError("beta must be non-negative for reward maximization.")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor.")
    if rewards.numel() == 0:
        raise ValueError("rewards must not be empty.")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contain NaN or Inf.")

    logits = beta * rewards
    return torch.softmax(logits - logits.max(), dim=0)


def effective_sample_size(probabilities: torch.Tensor) -> float:
    """Effective sample size of a normalized discrete distribution."""

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional tensor.")
    total = probabilities.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have positive finite mass.")
    p = probabilities / total
    return float((1.0 / p.square().sum()).item())


def first_order_gaussian(
    mean: torch.Tensor,
    variance: float,
    beta: float,
    reward_gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order Gaussian approximation for ``exp(+beta * reward)``."""

    _validate_local_inputs(mean, variance, beta, reward_gradient)
    batch, dim = mean.shape
    eye = torch.eye(dim, device=mean.device, dtype=mean.dtype)
    covariance = variance * eye.expand(batch, dim, dim).clone()
    corrected_mean = mean + beta * variance * reward_gradient
    return corrected_mean, covariance


def stabilize_hessian_for_precision(
    hessian: torch.Tensor,
    variance: float,
    beta: float,
    precision_floor: float = 1e-4,
) -> tuple[torch.Tensor, PrecisionDiagnostics]:
    """Cap only curvature that would make ``I - beta*variance*H`` singular.

    Positive reward curvature is mathematically valid while the local Gaussian
    remains normalizable, so it must not be discarded wholesale.  Eigenvalues
    are clipped only at the largest value compatible with ``precision_floor``;
    negative curvature is kept unchanged.
    """

    if hessian.ndim != 3 or hessian.shape[-1] != hessian.shape[-2]:
        raise ValueError("hessian must have shape (batch, dim, dim).")
    if variance < 0.0:
        raise ValueError("variance must be non-negative.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if not 0.0 < precision_floor <= 1.0:
        raise ValueError("precision_floor must lie in (0, 1].")

    symmetric = 0.5 * (hessian + hessian.transpose(-1, -2))
    if beta == 0.0 or variance == 0.0:
        return symmetric, PrecisionDiagnostics(1.0, 1.0, 0)

    evals, evecs = torch.linalg.eigh(symmetric.double())
    max_curvature = (1.0 - precision_floor) / (beta * variance)
    clipped = evals.clamp(max=max_curvature)
    stabilized = evecs @ torch.diag_embed(clipped) @ evecs.transpose(-1, -2)
    stabilized = stabilized.to(hessian.dtype)

    min_before = float((1.0 - beta * variance * evals.max(dim=-1).values).min().item())
    min_after = float((1.0 - beta * variance * clipped.max(dim=-1).values).min().item())
    num_clipped = int((evals > max_curvature).sum().item())
    return stabilized, PrecisionDiagnostics(min_before, min_after, num_clipped)


def second_order_gaussian(
    mean: torch.Tensor,
    variance: float,
    beta: float,
    reward_gradient: torch.Tensor,
    reward_hessian: torch.Tensor,
    precision_floor: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, PrecisionDiagnostics]:
    """Stable full-Hessian second-order Gaussian approximation.

    This routine is intended for low-dimensional validation/training paths.  It
    performs a batched dense solve and therefore must not be used on FLUX-sized
    latent tensors; FLUX uses the algebraically equivalent low-rank Woodbury
    path in :mod:`flux.pipeline`.
    """

    _validate_local_inputs(mean, variance, beta, reward_gradient)
    if reward_hessian.shape != (mean.shape[0], mean.shape[1], mean.shape[1]):
        raise ValueError("reward_hessian has an incompatible shape.")

    batch, dim = mean.shape
    if variance == 0.0:
        covariance = torch.zeros(batch, dim, dim, device=mean.device, dtype=mean.dtype)
        return mean.clone(), covariance, PrecisionDiagnostics(1.0, 1.0, 0)

    stable_hessian, diagnostics = stabilize_hessian_for_precision(
        reward_hessian, variance, beta, precision_floor
    )
    eye = torch.eye(dim, device=mean.device, dtype=mean.dtype).expand(batch, dim, dim)
    precision = eye / variance - beta * stable_hessian
    covariance = torch.linalg.solve(precision, eye)
    shift = beta * torch.bmm(covariance, reward_gradient.unsqueeze(-1)).squeeze(-1)
    return mean + shift, covariance, diagnostics


def second_order_preconditioned_gradient(
    reward_gradient: torch.Tensor,
    reward_hessian: torch.Tensor,
    variance: float,
    beta: float,
    precision_floor: float = 1e-4,
) -> tuple[torch.Tensor, PrecisionDiagnostics]:
    """Return ``(I - beta*variance*H)^-1 (beta*grad_r)``.

    This is the gradient form used by the Doob-guidance sampler.  Multiplying
    the result by ``variance`` gives the second-order Gaussian mean shift.
    """

    if reward_gradient.ndim != 2:
        raise ValueError("reward_gradient must have shape (batch, dim).")
    dummy_mean = torch.zeros_like(reward_gradient)
    _, covariance, diagnostics = second_order_gaussian(
        dummy_mean,
        variance,
        beta,
        reward_gradient,
        reward_hessian,
        precision_floor,
    )
    if variance == 0.0:
        return beta * reward_gradient, diagnostics
    preconditioned = beta * torch.bmm(
        covariance / variance, reward_gradient.unsqueeze(-1)
    ).squeeze(-1)
    return preconditioned, diagnostics


def stabilize_low_rank_curvatures(
    gram: torch.Tensor,
    curvatures: torch.Tensor,
    scale: float,
    precision_floor: float = 0.05,
    binary_search_steps: int = 32,
) -> tuple[torch.Tensor, LowRankPrecisionDiagnostics]:
    """Stabilize ``I - scale * W diag(curvatures) W.T`` in rank space.

    ``gram`` is ``W.T @ W``. Negative curvature always increases precision and
    is left unchanged. If positive curvature would cross the requested floor,
    all selected positive eigenvalues are damped by the largest common factor
    in ``[0, 1]`` that keeps the precision positive definite. This preserves
    valid positive curvature instead of silently deleting it.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix.")
    if curvatures.ndim != 1 or curvatures.shape[0] != gram.shape[0]:
        raise ValueError("curvatures must match the Gram-matrix dimension.")
    if scale < 0.0:
        raise ValueError("scale must be non-negative.")
    if not 0.0 < precision_floor <= 1.0:
        raise ValueError("precision_floor must lie in (0, 1].")

    if curvatures.numel() == 0 or scale == 0.0:
        diagnostics = LowRankPrecisionDiagnostics(1.0, 1.0, 1.0)
        return curvatures, diagnostics

    math_dtype = torch.float64
    symmetric_gram = 0.5 * (gram + gram.transpose(-1, -2))
    gram_evals, gram_evecs = torch.linalg.eigh(symmetric_gram.to(math_dtype))
    gram_evals = gram_evals.clamp_min(0.0)
    # R.T @ R = gram, which gives the non-zero precision spectrum in rank space.
    root = torch.diag(gram_evals.sqrt()) @ gram_evecs.transpose(-1, -2)
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=math_dtype)
    mu = curvatures.to(math_dtype)

    def minimum_precision(candidate: torch.Tensor) -> torch.Tensor:
        small_hessian = root @ torch.diag(candidate) @ root.transpose(-1, -2)
        return torch.linalg.eigvalsh(identity - scale * small_hessian).min()

    before = minimum_precision(mu)
    positive = mu.clamp_min(0.0)
    negative = mu.clamp_max(0.0)
    alpha = 1.0
    if before < precision_floor and torch.any(positive > 0):
        low, high = 0.0, 1.0
        for _ in range(binary_search_steps):
            middle = 0.5 * (low + high)
            if minimum_precision(negative + middle * positive) >= precision_floor:
                low = middle
            else:
                high = middle
        alpha = low

    stabilized = negative + alpha * positive
    after = minimum_precision(stabilized)
    diagnostics = LowRankPrecisionDiagnostics(
        min_eigenvalue_before=float(before.item()),
        min_eigenvalue_after=float(after.item()),
        positive_curvature_scale=float(alpha),
    )
    return stabilized.to(curvatures.dtype), diagnostics


def _validate_local_inputs(
    mean: torch.Tensor,
    variance: float,
    beta: float,
    reward_gradient: torch.Tensor,
) -> None:
    if mean.ndim != 2:
        raise ValueError("mean must have shape (batch, dim).")
    if reward_gradient.shape != mean.shape:
        raise ValueError("reward_gradient must have the same shape as mean.")
    if variance < 0.0:
        raise ValueError("variance must be non-negative.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative for reward maximization.")
    if not torch.isfinite(mean).all() or not torch.isfinite(reward_gradient).all():
        raise ValueError("mean/reward_gradient contains NaN or Inf.")
