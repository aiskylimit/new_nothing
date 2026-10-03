"""TROPIC_Proposal_v8, Section 4.5/4.7: allocating a FIXED TOTAL trust-region
budget across roll-out positions by leverage, instead of the uniform radius
v5-v7 (tropic/loss.py, tropic/loss_k.py, tropic/loss_pc.py) use. Pure tensor
math, no model/tokenizer dependency - mirrors tropic/primitives.py's own
convention so it can be unit-tested without ever touching a real model.
"""
from __future__ import annotations

import torch
from torch import Tensor


def token_entropy(log_old: Tensor) -> Tensor:
    """H(pi^old_t) = -sum_v p(v) log p(v), per position (Eq. 12's H term).
    log_old: [T, V], already log_softmax'd. Returns [T], always >= 0
    (Shannon entropy of a genuine probability distribution)."""
    p = log_old.exp()
    return -(p * log_old).sum(dim=-1)


def leverage(H: Tensor, abs_advantage_per_token: Tensor, rho: float, eta: float) -> Tensor:
    """Eq. 12: l_t = H(pi^old_t)^rho * (eta + |Ahat_j(t)|).

    `H`: [T], `token_entropy`'s output (>= 0, so H**rho is well-defined for
    any rho >= 0, including non-integer). `abs_advantage_per_token`: [T] =
    |Ahat_j(t)| already broadcast from per-step to per-token (pass the
    ABSOLUTE value - Eq. 12 uses |Ahat|, not the signed credit that the
    optional tilt ablation in tropic.loss_pc uses).

    eta -> large recovers ENTROPY-ONLY allocation (the value term becomes a
    near-constant additive offset, so l_t is governed by H alone regardless
    of Ahat - see tests/test_leverage.py). rho=0 recovers VALUE-ONLY
    allocation (H**0 == 1 everywhere). l_t === eta*(uniform) with rho=0 and
    abs_advantage_per_token===0 recovers the uniform radius of v5-v7 exactly
    (constant leverage -> allocate_epsilon returns eps_mean everywhere).
    """
    return H.clamp(min=0.0).pow(rho) * (eta + abs_advantage_per_token)


def allocate_epsilon(leverage_t: Tensor, eps_mean: float, eps_max: float, seq_len: int | None = None) -> Tensor:
    """Eq. 13: epsilon_t = min(seq_len*eps_mean * l_t / sum(l), eps_max),
    with ONE redistribution pass of whatever budget the cap frees, spent
    proportionally to leverage among the still-uncapped positions - so that
    sum(epsilon_t) <= seq_len*eps_mean always holds (Proposition 8: this is
    exactly what keeps every sequence-level drift guarantee unchanged).

    `seq_len` defaults to `leverage_t.numel()` (allocate over the SAME
    roll-out `leverage_t` was computed on - the common case). Pass it
    explicitly only if `leverage_t` covers a strict sub-sequence.
    """
    T = leverage_t.shape[0] if seq_len is None else seq_len
    total_budget = T * eps_mean
    total_leverage = leverage_t.sum()
    if total_leverage <= 0:
        # Degenerate (T==0, or every leverage value is exactly zero) - fall
        # back to the uniform radius rather than a 0/0 division.
        return torch.full_like(leverage_t, eps_mean)

    eps_t = total_budget * leverage_t / total_leverage
    capped = eps_t > eps_max
    if capped.any():
        eps_t = torch.where(capped, torch.full_like(eps_t, eps_max), eps_t)
        freed = total_budget - eps_t.sum()
        uncapped = ~capped
        if freed > 0 and uncapped.any():
            uncapped_leverage_sum = leverage_t[uncapped].sum()
            if uncapped_leverage_sum > 0:
                bonus = freed * leverage_t[uncapped] / uncapped_leverage_sum
                eps_t = eps_t.clone()
                eps_t[uncapped] = eps_t[uncapped] + bonus
                # Eq. 13 calls for exactly ONE redistribution pass, not a
                # fixed-point loop - if that single pass pushes a
                # previously-uncapped position over eps_max, cap it again
                # here (the numerical safety net below still guarantees the
                # total budget is never exceeded either way).
                eps_t = torch.clamp(eps_t, max=eps_max)

    # Numerical safety net: if float error ever pushes the sum a hair above
    # the budget, rescale down proportionally (never enlarges it, and is a
    # no-op to float precision in the common uncapped case).
    total_now = eps_t.sum()
    if total_now > total_budget:
        eps_t = eps_t * (total_budget / total_now)
    return eps_t
