"""Pure-tensor math primitives for TROPIC.

Everything here operates on log-probabilities of shape [..., V] (already
log_softmax'd along the last dim) unless noted otherwise, and is independent
of any language model. This lets the core algorithm (Eq. 8-19 of
TROPIC_Version_Control.pdf) be unit-tested without ever touching Qwen2.5.

Naming follows the paper:
    log_teacher  -> log pi^T_t   (or log pi-bar^T_t after debiasing)
    log_old      -> log pi^old_t (checkpoint that produced the roll-out)
    log_student  -> log pi^S_t   (trainable)
    log_pi_eps   -> log pi^eps_t (projected target, Eq. 14)
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Basic KL on categorical distributions given as log-probabilities.
# ---------------------------------------------------------------------------


def kl_categorical(log_p: Tensor, log_q: Tensor, dim: int = -1) -> Tensor:
    """KL(p ‖ q) = sum_v p(v) (log p(v) - log q(v))."""
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=dim)


def normalize_log(logits: Tensor, dim: int = -1) -> Tensor:
    """log_softmax, exposed under a name that matches the paper's ⌊·⌋-free algebra."""
    return logits - torch.logsumexp(logits, dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# Eq. 11 - classifier-free guidance debiasing of the teacher (alpha=0 -> no-op).
# ---------------------------------------------------------------------------


def debias_logits(
    logits_with_context: Tensor,
    logits_context_only: Tensor | None,
    alpha: float = 0.0,
) -> Tensor:
    """z_theta(.|x,o<t,r) - alpha * z_theta(.|empty,o<t,r), pre-softmax.

    For TROPIC-P (alpha=0) this is the identity and the caller should never
    have computed `logits_context_only` in the first place (skip the extra
    forward pass entirely) - this function only exists so TROPIC-G can reuse
    the exact same code path later.
    """
    if alpha == 0.0 or logits_context_only is None:
        return logits_with_context
    return logits_with_context - alpha * logits_context_only


# ---------------------------------------------------------------------------
# Eq. 14-15, Proposition 2 - closed-form projection onto the KL ball.
# ---------------------------------------------------------------------------


def build_log_pi_eps(s: Tensor, log_teacher: Tensor, log_old: Tensor) -> Tensor:
    """log pi^eps_s(v) = s*log_teacher(v) + (1-s)*log_old(v) - logZ_s, Eq. 14."""
    if s.dim() == log_teacher.dim() - 1:
        s = s.unsqueeze(-1)
    unnorm = s * log_teacher + (1.0 - s) * log_old
    return normalize_log(unnorm)


def phi(s: Tensor, log_teacher: Tensor, log_old: Tensor) -> Tensor:
    """phi(s) = KL(pi^eps_s ‖ pi^old), Eq. 15. Strictly increasing, phi(0)=0."""
    log_pi_eps = build_log_pi_eps(s, log_teacher, log_old)
    return kl_categorical(log_pi_eps, log_old)


def bisect_step_size(
    log_teacher: Tensor,
    log_old: Tensor,
    epsilon: float,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Tensor:
    """Solve phi(s) = epsilon for s in (0,1], per leading-dim position.

    log_teacher, log_old: [..., V]. Returns s: [...] (one scalar per position).
    s=1 exactly when the constraint is already idle (KL(teacher||old) <= eps).
    """
    del tol  # fixed-iteration bisection; 60 halvings is far below any float tol
    kl_at_1 = kl_categorical(log_teacher, log_old)
    idle = kl_at_1 <= epsilon

    s = torch.ones_like(kl_at_1)
    active = ~idle
    if active.any():
        log_teacher_a = log_teacher[active]
        log_old_a = log_old[active]
        lo = torch.zeros_like(kl_at_1[active])
        hi = torch.ones_like(kl_at_1[active])
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = phi(mid, log_teacher_a, log_old_a)
            too_high = val > epsilon
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)
        s_active = 0.5 * (lo + hi)
        s[active] = s_active
    return s


def project_to_kl_ball(
    log_teacher: Tensor,
    log_old: Tensor,
    epsilon: float,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> tuple[Tensor, Tensor]:
    """Proposition 2: closed-form projection of the (debiased) teacher onto B_t.

    Returns (log_pi_eps, s). Caller is responsible for `.detach()`-ing the
    inputs beforehand (this function does not carry gradient by design: the
    projected object is the *target*, never the policy being optimized).
    """
    s = bisect_step_size(log_teacher, log_old, epsilon, tol, max_iter)
    log_pi_eps = build_log_pi_eps(s, log_teacher, log_old)
    return log_pi_eps, s


def debias_and_project(
    logits_with_context: Tensor,
    logits_context_only: Tensor | None,
    log_old: Tensor,
    alpha: float,
    epsilon: float,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> tuple[Tensor, Tensor]:
    """Two-step reference implementation: debias (Eq. 11) then project (Eq. 13-14).

    Used by TROPIC-G, and by tests/test_primitives.py to check that this
    matches the one-shot log-linear composition of Eq. 16.
    """
    logits_debiased = debias_logits(logits_with_context, logits_context_only, alpha)
    log_teacher_bar = normalize_log(logits_debiased)
    return project_to_kl_ball(log_teacher_bar, log_old, epsilon, tol, max_iter)


def compose_log_linear(
    logits_with_context: Tensor,
    logits_context_only: Tensor | None,
    log_old: Tensor,
    alpha: float,
    s: Tensor,
) -> Tensor:
    """Eq. 16: z^eps_t = s*z(x,o<t,r) - s*alpha*z(empty,o<t,r) + (1-s)*log_old.

    Takes a *given* step size `s` (e.g. computed by debias_and_project) and
    builds the target directly from the three cached logit tensors in one
    log-linear combination, with coefficients (s, -s*alpha, 1-s).
    """
    if s.dim() == log_old.dim() - 1:
        s = s.unsqueeze(-1)
    ctx_only_term = 0.0 if (alpha == 0.0 or logits_context_only is None) else s * alpha * logits_context_only
    z_eps = s * logits_with_context - ctx_only_term + (1.0 - s) * log_old
    return normalize_log(z_eps)


# ---------------------------------------------------------------------------
# Eq. 17, Proposition 4 - student-skewed reverse KL.
# ---------------------------------------------------------------------------


def skewed_reverse_kl(log_student: Tensor, log_target: Tensor, beta: float) -> Tensor:
    """D^(beta)(p ‖ q) = KL(p ‖ beta*p + (1-beta)*q), Eq. 17.

    Computed in log-space via logsumexp for numerical stability; recovers the
    ordinary reverse KL as beta -> 0 and is bounded above by log(1/beta)
    (Proposition 4) for any beta in (0, 1].
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    log_beta = math.log(beta)
    log_1mb = math.log1p(-beta) if beta < 1.0 else float("-inf")
    stacked = torch.stack([log_beta + log_student, log_1mb + log_target], dim=0)
    log_m = torch.logsumexp(stacked, dim=0)
    p = log_student.exp()
    return (p * (log_student - log_m)).sum(dim=-1)


def skewed_ratio_bound_check(log_student: Tensor, log_target: Tensor, beta: float) -> Tensor:
    """Per-token likelihood ratio p(v)/(beta*p(v)+(1-beta)*q(v)); must be <= 1/beta."""
    log_beta = math.log(beta)
    log_1mb = math.log1p(-beta) if beta < 1.0 else float("-inf")
    stacked = torch.stack([log_beta + log_student, log_1mb + log_target], dim=0)
    log_m = torch.logsumexp(stacked, dim=0)
    return (log_student - log_m).exp()


# ---------------------------------------------------------------------------
# Sparsification: shared top-K mask + default tail mass (Remark 4).
# ---------------------------------------------------------------------------


def build_shared_topk_mask(log_ref: Tensor, sampled_token_ids: Tensor, top_k: int) -> Tensor:
    """Fixed-width [..., top_k] index tensor: top-K tokens by `log_ref` mass,
    always including `sampled_token_ids` (swapped into the lowest-mass slot
    if it would otherwise be dropped). Reuse this SAME index tensor for every
    distribution (pi^S, pi^old, pi-bar^T, pi^eps) - that is what Remark 4
    requires ("all distributions share a single mask")."""
    probs = log_ref.exp()
    top_k = min(top_k, probs.shape[-1])
    _, topk_idx = torch.topk(probs, k=top_k, dim=-1)
    in_mask = (topk_idx == sampled_token_ids.unsqueeze(-1)).any(dim=-1)
    missing = ~in_mask
    if missing.any():
        topk_idx = topk_idx.clone()
        topk_idx[missing, -1] = sampled_token_ids[missing]
    return topk_idx


def apply_sparse_mask(log_p: Tensor, mask_indices: Tensor, default_mass: float) -> Tensor:
    """Gather `log_p` onto `mask_indices`, add one tail bucket carrying at
    least `default_mass`, and renormalize. Output is [..., K+1] log-probs
    (differentiable w.r.t. log_p, so this may be applied to log_student too).
    """
    gathered_log_p = torch.gather(log_p, dim=-1, index=mask_indices)
    gathered_p = gathered_log_p.exp()
    kept_mass = gathered_p.sum(dim=-1, keepdim=True)
    tail_mass = torch.clamp(1.0 - kept_mass, min=0.0)
    tail_prob = torch.maximum(tail_mass, torch.full_like(tail_mass, default_mass))
    sparse_probs = torch.cat([gathered_p, tail_prob], dim=-1)
    sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True)
    return torch.log(sparse_probs.clamp_min(1e-12))


def total_variation(log_p: Tensor, log_q: Tensor, dim: int = -1) -> Tensor:
    """TV(p, q) = 0.5 * sum_v |p(v) - q(v)|."""
    return 0.5 * (log_p.exp() - log_q.exp()).abs().sum(dim=dim)


# ---------------------------------------------------------------------------
# TROPIC_Proposal_v8, Eq. 13/14 - leverage-allocated trust region: same
# projection as bisect_step_size/project_to_kl_ball above, except epsilon is
# now a PER-POSITION Tensor[T] (tropic.leverage.allocate_epsilon's output)
# instead of one scalar shared by the whole roll-out. Added here rather than
# generalizing the scalar functions in place so every existing caller
# (tropic.loss.tropic_p_loss, tropic.loss_pc.tropic_pc_loss) keeps calling
# the untouched scalar path byte-for-byte.
# ---------------------------------------------------------------------------


def bisect_step_size_vec(
    log_teacher: Tensor,
    log_old: Tensor,
    epsilon: Tensor,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Tensor:
    """Like `bisect_step_size`, but `epsilon` is a Tensor[T] (one radius per
    leading-dim position) instead of a python float. log_teacher, log_old:
    [T, V]. Returns s: [T]. Idle (s=1) is decided PER POSITION: position t is
    idle iff KL(teacher_t || old_t) <= epsilon[t]."""
    del tol  # fixed-iteration bisection, same convention as bisect_step_size
    kl_at_1 = kl_categorical(log_teacher, log_old)
    idle = kl_at_1 <= epsilon

    s = torch.ones_like(kl_at_1)
    active = ~idle
    if active.any():
        log_teacher_a = log_teacher[active]
        log_old_a = log_old[active]
        epsilon_a = epsilon[active]
        lo = torch.zeros_like(kl_at_1[active])
        hi = torch.ones_like(kl_at_1[active])
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = phi(mid, log_teacher_a, log_old_a)
            too_high = val > epsilon_a
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)
        s_active = 0.5 * (lo + hi)
        s[active] = s_active
    return s


def project_to_kl_ball_vec(
    log_teacher: Tensor,
    log_old: Tensor,
    epsilon: Tensor,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> tuple[Tensor, Tensor]:
    """Vector-epsilon counterpart of `project_to_kl_ball` (Proposition 1 of
    v8: same closed form, per-position radius epsilon_t from Eq. 13)."""
    s = bisect_step_size_vec(log_teacher, log_old, epsilon, tol, max_iter)
    log_pi_eps = build_log_pi_eps(s, log_teacher, log_old)
    return log_pi_eps, s
