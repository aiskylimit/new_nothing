"""Phase 5: the TROPIC-P/TROPIC-G objective - Eq. 18 with mu=0 (no hinge
certificate). TROPIC-P is the alpha=0 special case (no debiasing forward
pass); TROPIC-G (alpha>0) additionally debiases the teacher (Eq. 11) before
projecting. Only the sparsified skewed-reverse-KL distillation term against
the projected target pi^eps_t remains either way.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tropic.primitives import (
    apply_sparse_mask,
    build_shared_topk_mask,
    debias_logits,
    normalize_log,
    project_to_kl_ball,
    skewed_reverse_kl,
)


def tropic_p_loss(
    log_student: Tensor,
    log_teacher: Tensor,
    log_old: Tensor,
    generated_ids: Tensor,
    epsilon: float,
    beta: float,
    top_k: int,
    default_mass: float,
    alpha: float = 0.0,
    teacher_context_only_logits: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """log_student: [T, V] WITH grad. log_teacher, log_old: [T, V] detached.

    `alpha`/`teacher_context_only_logits`: TROPIC-**G** only (default
    alpha=0.0 recovers TROPIC-P exactly, unchanged from before). When
    alpha>0, `log_teacher` must be RAW LOGITS (not log-softmax - see
    `tropic.model.ContextualPolicy.forward_teacher_logits`), matching
    `teacher_context_only_logits` (`forward_teacher_context_only`'s raw-logit
    output) - both get combined via Eq. 11's `debias_logits` BEFORE
    normalizing, since subtracting two already-normalized log_softmax outputs
    is not equivalent (each carries its own normalization constant). When
    alpha=0.0, `log_teacher` is used exactly as before (log_softmax already
    applied by the caller) - `debias_logits` is the identity in that case
    anyway, so this dual convention is safe.

    Returns (scalar loss, step_size s [T]) - `s` is returned purely for
    diagnostics/logging (Section 4.5's "readable diagnostic").
    """
    assert log_teacher.requires_grad is False, "teacher must be cached/detached (Section 4.5)"
    assert log_old.requires_grad is False, "checkpoint must be cached/detached (Section 4.5)"
    log_teacher = log_teacher.detach()
    log_old = log_old.detach()

    if alpha > 0.0:
        assert teacher_context_only_logits is not None, (
            "TROPIC-G (alpha>0) requires teacher_context_only_logits (Eq. 11's second term)"
        )
        assert teacher_context_only_logits.requires_grad is False, (
            "teacher_context_only_logits must be cached/detached (Section 4.5), same contract as log_teacher"
        )
        teacher_context_only_logits = teacher_context_only_logits.detach()
        log_teacher = normalize_log(debias_logits(log_teacher, teacher_context_only_logits, alpha))

    # Remark 4: build ONE shared top-K mask (from the checkpoint distribution,
    # which is always well-defined and never near-degenerate the way the raw
    # teacher can be) and always keep the sampled token.
    mask = build_shared_topk_mask(log_old, generated_ids, top_k)

    log_student_sp = apply_sparse_mask(log_student, mask, default_mass)
    log_teacher_sp = apply_sparse_mask(log_teacher, mask, default_mass)
    log_old_sp = apply_sparse_mask(log_old, mask, default_mass)

    # Eq. 13-14: project the (debiased when alpha>0, identity when alpha=0)
    # teacher onto the KL ball around the checkpoint. This carries no
    # gradient by construction - project_to_kl_ball only ever receives
    # detached inputs.
    log_pi_eps, s = project_to_kl_ball(log_teacher_sp, log_old_sp, epsilon)
    log_pi_eps = log_pi_eps.detach()
    assert log_pi_eps.requires_grad is False

    # Eq. 17: student-skewed reverse KL against the projected waypoint.
    per_token = skewed_reverse_kl(log_student_sp, log_pi_eps, beta)
    loss = per_token.mean()
    return loss, s


@torch.no_grad()
def privileged_gap(log_old: Tensor, log_teacher: Tensor) -> Tensor:
    """G_k = E[TV(pi^old_t, pi-bar^T_t)], Eq. 19 - logged for diagnostics only
    (TROPIC-P does not use it to control epsilon; that is TROPIC-G's job)."""
    from tropic.primitives import total_variation

    return total_variation(log_old, log_teacher).mean()
