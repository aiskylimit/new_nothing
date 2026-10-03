"""TROPIC-PC: Sec 4.5/4.8's process-tilted objective (Eq. 12, 15, 17) on top
of TROPIC-G/TROPIC-K's existing debias -> project -> skewed-KL pipeline
(tropic/loss.py, tropic/loss_k.py - neither file is imported from or
modified; this module duplicates their sparsification/projection wiring the
same way loss_k.py already duplicates loss.py's, so each file stays fully
decoupled and existing callers are unaffected).

Pipeline, in order (Sec 4.5's own "why the tilt goes into the target"):
  1. debias   (Eq. 9/11, alpha>0 only - identical to loss_k.py)
  2. sparsify (Remark 4 - ONE shared top-K mask for student/teacher/old)
  3. tilt     (Eq. 12 - kappa*Ahat_j(t) added to the SAMPLED token's logit,
               valid under sparsification because build_shared_topk_mask
               always keeps the sampled token in the mask - Remark 4)
  4. project  (Eq. 13/14, onto the KL ball around the checkpoint)
  5. skewed KL (Eq. 17, weighted by w_t; optional mu-hinge, Eq. 18)

At `tilt` all-zero (or None), `token_weights=None` and `mu=0.0`, this
reproduces tropic_p_loss/tropic_k_loss's (loss, s) BIT-FOR-BIT - the tilt
step is skipped ENTIRELY (not "added zero then renormalized") specifically
so kappa=0 is an exact no-op, not merely a floating-point-close one.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tropic.primitives import (
    apply_sparse_mask,
    build_shared_topk_mask,
    debias_logits,
    kl_categorical,
    normalize_log,
    project_to_kl_ball,
    skewed_reverse_kl,
)


def tilt_sampled_token(log_p: Tensor, token_ids: Tensor, c) -> Tensor:
    """Lemma 2's single-token tilt, batched over the leading dimension:
    qc(v) ∝ q(v) * exp(c * 1[v=token_ids]) per row, renormalized. `c` may be
    a python float (same tilt every row) or a [T] tensor (per-row tilt, e.g.
    kappa*Ahat_j(t) from `tropic.process_credit.tilt_per_token`). A no-op
    (returns `log_p` unchanged) when every `c` is exactly zero, so callers
    get bit-for-bit kappa=0 behavior for free."""
    if not torch.is_tensor(c):
        c = torch.full((log_p.shape[0],), float(c), dtype=log_p.dtype, device=log_p.device)
    if not bool(torch.any(c != 0)):
        return log_p
    tilted = log_p.clone()
    idx = torch.arange(log_p.shape[0], device=log_p.device)
    tilted[idx, token_ids] = tilted[idx, token_ids] + c
    return normalize_log(tilted)


def tropic_pc_loss(
    log_student: Tensor,
    log_teacher: Tensor,
    log_old: Tensor,
    generated_ids: Tensor,
    epsilon: float,
    beta: float,
    top_k: int,
    default_mass: float,
    tilt: Tensor | None = None,
    token_weights: Tensor | None = None,
    alpha: float = 0.0,
    teacher_context_only_logits: Tensor | None = None,
    mu: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """log_student: [T, V] WITH grad. log_teacher, log_old: [T, V] detached.
    `tilt`: [T] = kappa*Ahat_j(t) (tropic.process_credit.tilt_per_token);
    None or all-zero recovers TROPIC-G/TROPIC-K's target exactly (kappa=0).
    `token_weights`: [T] = w_t (tropic.process_credit.first_error_weights);
    None recovers the uniform w_t=1 case exactly (no multiply is performed).
    `alpha`/`teacher_context_only_logits`/`mu`: same contract as
    tropic.loss.tropic_p_loss / tropic.loss_k.tropic_k_loss.

    Returns (loss, s, hinge_mean) - same shapes/semantics as tropic_k_loss.
    """
    assert log_teacher.requires_grad is False, "teacher must be cached/detached (Section 4.5)"
    assert log_old.requires_grad is False, "checkpoint must be cached/detached (Section 4.5)"
    log_teacher = log_teacher.detach()
    log_old = log_old.detach()

    if alpha > 0.0:
        assert teacher_context_only_logits is not None, (
            "alpha>0 (TROPIC-G debiasing) requires teacher_context_only_logits"
        )
        assert teacher_context_only_logits.requires_grad is False, (
            "teacher_context_only_logits must be cached/detached (Section 4.5), same contract as log_teacher"
        )
        teacher_context_only_logits = teacher_context_only_logits.detach()
        log_teacher = normalize_log(debias_logits(log_teacher, teacher_context_only_logits, alpha))

    # Remark 4: one shared top-K mask for every distribution, including the
    # tilt and hinge terms below.
    mask = build_shared_topk_mask(log_old, generated_ids, top_k)

    log_student_sp = apply_sparse_mask(log_student, mask, default_mass)
    log_teacher_sp = apply_sparse_mask(log_teacher, mask, default_mass)
    log_old_sp = apply_sparse_mask(log_old, mask, default_mass)

    # Eq. 12: tilt the SAMPLED token's logit in the (debiased) teacher by
    # kappa*Ahat_j(t), then renormalize - safe under sparsification because
    # build_shared_topk_mask guarantees the sampled token is always in
    # `mask` (Remark 4's own note: "the sampled token is always in the
    # mask"). Skipped ENTIRELY when every tilt value is zero, so kappa=0 is
    # bit-for-bit identical to the untilted (TROPIC-G/-K) target, not just
    # numerically close after a no-op renormalization.
    if tilt is not None:
        sampled_col = (mask == generated_ids.unsqueeze(-1)).float().argmax(dim=-1)  # [T]
        log_teacher_sp = tilt_sampled_token(log_teacher_sp, sampled_col, tilt)

    # Eq. 13-14: project the (debiased, tilted) teacher onto the KL ball
    # around the checkpoint. No gradient by construction.
    log_pi_eps, s = project_to_kl_ball(log_teacher_sp, log_old_sp, epsilon)
    log_pi_eps = log_pi_eps.detach()
    assert log_pi_eps.requires_grad is False

    # Eq. 17: w_t * ( D^(beta)(pi^S || pi^eps) + mu*[KL(pi^S||pi^old)-eps]_+ ).
    per_token_distill = skewed_reverse_kl(log_student_sp, log_pi_eps, beta)
    kl_student_old = kl_categorical(log_student_sp, log_old_sp.detach())
    hinge_per_token = torch.clamp(kl_student_old - epsilon, min=0.0)
    per_token_total = per_token_distill + mu * hinge_per_token
    if token_weights is not None:
        per_token_total = per_token_total * token_weights
    loss = per_token_total.mean()

    return loss, s, hinge_per_token.mean().detach()


# ---------------------------------------------------------------------------
# Reference (unsparsified) three-step construction and Eq. 15's one-shot
# log-linear composition - used only by tests/test_process_credit.py to
# check the two agree, mirroring how tropic/primitives.py's
# debias_and_project/compose_log_linear are checked against each other in
# tests/test_primitives.py.
# ---------------------------------------------------------------------------


def debias_tilt_and_project(
    logits_with_context: Tensor,
    logits_context_only: Tensor | None,
    log_old: Tensor,
    generated_ids: Tensor,
    alpha: float,
    tilt: Tensor | None,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Three-step reference: debias (Eq. 9/11) -> tilt (Eq. 12) -> project
    (Eq. 13/14), on the FULL vocabulary (no sparsification)."""
    logits_debiased = debias_logits(logits_with_context, logits_context_only, alpha)
    log_teacher_bar = normalize_log(logits_debiased)
    log_teacher_tilde = log_teacher_bar if tilt is None else tilt_sampled_token(log_teacher_bar, generated_ids, tilt)
    return project_to_kl_ball(log_teacher_tilde, log_old, epsilon)


def compose_log_linear_pc(
    logits_with_context: Tensor,
    logits_context_only: Tensor | None,
    log_old: Tensor,
    alpha: float,
    tilt: Tensor,
    generated_ids: Tensor,
    s: Tensor,
) -> Tensor:
    """Eq. 15: z^eps_t = s*z_phi(.|x,o<t,r) - s*alpha*z_phi(.|empty,o<t,r)
    + s*kappa*Ahat_j(t)*e_ot + (1-s)*log_old, given a precomputed step size
    `s` (e.g. from `debias_tilt_and_project`). Extends
    `tropic.primitives.compose_log_linear` with the tilt term, which is also
    linear in log-space at the sampled-token position."""
    if s.dim() == log_old.dim() - 1:
        s = s.unsqueeze(-1)
    ctx_only_term = 0.0 if (alpha == 0.0 or logits_context_only is None) else s * alpha * logits_context_only
    z_eps = s * logits_with_context - ctx_only_term + (1.0 - s) * log_old
    if tilt is not None and bool(torch.any(tilt != 0)):
        T = z_eps.shape[0]
        tilt_term = torch.zeros_like(z_eps)
        tilt_term[torch.arange(T, device=z_eps.device), generated_ids] = s.squeeze(-1) * tilt
        z_eps = z_eps + tilt_term
    return normalize_log(z_eps)
