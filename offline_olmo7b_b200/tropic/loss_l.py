"""tropic/loss_l.py: Section 4.5-4.9 of TROPIC_Proposal_v8 - debias (Eq. 9)
-> optional tilt (kappa ablation, v6/v7's Eq. 14 mechanism, default off) ->
project onto a PER-POSITION allocated trust region (Eq. 13/17,
tropic.leverage.allocate_epsilon's output, via tropic.primitives.
project_to_kl_ball_vec) -> skewed reverse KL (Eq. 18), optional hinge
(Eq. 19). Duplicates loss.py/loss_k.py/loss_pc.py's sparsification wiring
rather than importing the loss functions from them - same convention those
files already established, so this file has zero coupling to any of them
and none of them is modified.

With `epsilon_t` CONSTANT across positions (v8's alloc="uniform") and
`tilt=None` (kappa=0, v8's own new default - "tilt off"), this reproduces
tropic_p_loss's (loss, s) bit-for-bit - see tests/test_loss_l.py. The tilt
step is skipped ENTIRELY when `tilt` is None/all-zero, the same convention
tropic.loss_pc.tropic_pc_loss already uses (reused here via
`tilt_sampled_token`, not reimplemented).
"""
from __future__ import annotations

import torch
from torch import Tensor

from tropic.loss_pc import tilt_sampled_token
from tropic.primitives import (
    apply_sparse_mask,
    build_shared_topk_mask,
    debias_logits,
    kl_categorical,
    normalize_log,
    project_to_kl_ball_vec,
    skewed_reverse_kl,
)


def tropic_l_loss(
    log_student: Tensor,
    log_teacher: Tensor,
    log_old: Tensor,
    generated_ids: Tensor,
    epsilon_t: Tensor,
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

    `epsilon_t`: [T] per-position trust-region radius - pass
    `tropic.leverage.allocate_epsilon`'s output for entropy/value/leverage
    allocation, or a constant-valued [T] tensor (e.g. `torch.full((T,),
    eps)`) to recover v5-v7's UNIFORM radius exactly - the loss itself does
    not know or care which allocation policy produced it.

    `tilt`/`token_weights`/`alpha`/`teacher_context_only_logits`/`mu`: same
    contract as `tropic.loss_pc.tropic_pc_loss` (v8's default is `tilt=None`
    i.e. kappa=0, "tilt off" - Table 1's new defaults; the tilt ablation of
    Section 4.6/Hypothesis 3 is still available by passing a nonzero tilt).

    Returns (loss, s, hinge_mean) - same shapes/semantics as
    tropic_pc_loss/tropic_k_loss.
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

    # Remark 6 (v8) / Remark 4 (v6-v7): one shared top-K mask for every
    # distribution, including the tilt and hinge terms below.
    mask = build_shared_topk_mask(log_old, generated_ids, top_k)

    log_student_sp = apply_sparse_mask(log_student, mask, default_mass)
    log_teacher_sp = apply_sparse_mask(log_teacher, mask, default_mass)
    log_old_sp = apply_sparse_mask(log_old, mask, default_mass)

    # Eq. 14 (ablation only, default off): tilt the sampled token's logit by
    # kappa*Ahat_j(t), skipped entirely when tilt is None/all-zero.
    if tilt is not None:
        sampled_col = (mask == generated_ids.unsqueeze(-1)).float().argmax(dim=-1)  # [T]
        log_teacher_sp = tilt_sampled_token(log_teacher_sp, sampled_col, tilt)

    # Eq. 13/15-17: project onto the ALLOCATED (per-position) KL ball -
    # Proposition 1's closed form, with epsilon replaced by epsilon_t.
    log_pi_eps, s = project_to_kl_ball_vec(log_teacher_sp, log_old_sp, epsilon_t)
    log_pi_eps = log_pi_eps.detach()
    assert log_pi_eps.requires_grad is False

    # Eq. 19: w_t * ( D^(beta)(pi^S||pi^eps) + mu*[KL(pi^S||pi^old)-eps_t]_+ ).
    per_token_distill = skewed_reverse_kl(log_student_sp, log_pi_eps, beta)
    kl_student_old = kl_categorical(log_student_sp, log_old_sp.detach())
    hinge_per_token = torch.clamp(kl_student_old - epsilon_t, min=0.0)
    per_token_total = per_token_distill + mu * hinge_per_token
    if token_weights is not None:
        per_token_total = per_token_total * token_weights
    loss = per_token_total.mean()

    return loss, s, hinge_per_token.mean().detach()
