"""OPSD's own objective, reimplemented on top of the SAME data/model/rollout
pipeline as TROPIC-P (tropic/data.py, tropic/model.py, tropic/rollout.py) so
the two methods differ ONLY in the loss - everything else (prompts, rollout
sampling, teacher/student/checkpoint forward passes) is shared, which is
what makes the eventual comparison table fair.

This module is a line-for-line port of `OPSDTrainer.generalized_jsd_loss`
(opsd_trainer.py, fetched live from github.com/siyan-zhao/OPSD on
2026-09-02 - the file could not be read in an earlier pass of this project,
which led to a formula-faithful-but-not-identical reconstruction; this
replaces that with the verbatim logic). Confirmed against the real launch
scripts (scripts/run_opsd_1b.sh et al.): every published run uses `beta=0`
(pure forward KL(teacher||student), NOT the beta=0.5 generalized-JSD mixture
assumed in an earlier version of this file) and never passes `--top_k_loss`,
i.e. the loss is computed over the FULL vocabulary - no top-k sparsification.
Sparsification (top_k/default_mass) is TROPIC's OWN mechanism (Remark 4 of
TROPIC_Version_Control.pdf), not OPSD's; keeping it in the OPSD baseline was
the earlier mistake this fixes.

Generalized JSD (Agarwal et al. 2024, arXiv:2306.13649), OPSD's exact formula:
    M(beta) = (1-beta) * pi^S + beta * pi^T      (mixture; beta multiplies the
                                                    TEACHER - opsd_trainer.py's
                                                    `teacher_log_probs + log(beta)`,
                                                    `student_log_probs + log1p(-beta)`)
    JSD_beta(pi^T, pi^S) = beta * KL(pi^T || M) + (1-beta) * KL(pi^S || M)
    beta=0  -> KL(pi^T || pi^S)   ("forward KL", OPSD's own default/only setting)
    beta=1  -> KL(pi^S || pi^T)   ("reverse KL")
Contrast with TROPIC's D^(beta)(p||q) = KL(p || beta*p + (1-beta)*q), which
mixes the STUDENT itself into its own denominator and is always mode-seeking
(Section 4.4 of the paper) - the two are deliberately different objectives,
not the same formula under a different name.

`generalized_jsd_loss` additionally applies TEMPERATURE SCALING to both
distributions before computing the divergence (`student_logits / temperature`
in the real code) - since `log_softmax(logits/T) == log_softmax(log_probs/T)`
(dividing by T then re-normalizing is invariant to whatever additive shift
turned the original logits into log-probs), this is applied here directly on
the already-log-softmaxed log_student/log_teacher tensors this codebase's
`ContextualPolicy._score` produces, with no need to plumb raw logits through.

CLIP ORDER (easy to get backwards - this file previously did): OPSD's own
README states it explicitly - "clipping is applied point-wise, not to the
vocabulary-summed loss" - i.e. `jsd_token_clip` caps each individual
(position, vocab-token) divergence CONTRIBUTION before summing over the
vocabulary, not the aggregate per-position divergence after the sum. Their
own code confirms this: `F.kl_div(..., reduction="none")` returns an
unreduced [batch, seq, vocab] tensor, `.clamp(...)` is applied to THAT, and
only the final `jsd.sum() / mask.sum()` at the very end performs the
vocab-sum-then-token-mean. Clipping the already-vocab-summed value (as an
earlier version of this file did) is a materially different, much looser
operation. This is also why OPSD's own docs note the loss "can be negative"
under clipping - independently capping same-sign summands can break the
cancellation that guarantees a true KL sum is non-negative.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def generalized_jsd_pointwise(log_student: Tensor, log_teacher: Tensor, beta: float) -> Tensor:
    """UNSUMMED per-(position, vocab-token) JSD_beta contribution, shape
    [T, V] - deliberately not reduced over the vocab dimension, so that
    `jsd_token_clip` can be applied point-wise first (see module docstring).
    Degenerate beta=0/1 cases match opsd_trainer.py's own special-casing
    (an exact KL, not a mixture with a vanishing weight) rather than the
    general formula's limit, avoiding a log(0) in `math.log(beta)`."""
    if beta == 0.0:
        return log_teacher.exp() * (log_teacher - log_student)  # pointwise KL(teacher || student)
    if beta == 1.0:
        return log_student.exp() * (log_student - log_teacher)  # pointwise KL(student || teacher)

    beta_t = torch.as_tensor(beta, dtype=log_student.dtype, device=log_student.device)
    log_mixture = torch.logsumexp(
        torch.stack([log_student + torch.log1p(-beta_t), log_teacher + torch.log(beta_t)], dim=0), dim=0
    )
    kl_teacher_pt = log_teacher.exp() * (log_teacher - log_mixture)
    kl_student_pt = log_student.exp() * (log_student - log_mixture)
    return beta * kl_teacher_pt + (1.0 - beta) * kl_student_pt


def opsd_baseline_loss(
    log_student: Tensor,
    log_teacher: Tensor,
    generated_ids: Tensor,
    beta: float = 0.0,
    jsd_token_clip: float | None = 0.05,
    temperature: float = 1.1,
    top_k: int | None = None,
    default_mass: float | None = None,
) -> Tensor:
    """OPSD's real training loss: temperature-scaled generalized JSD, clipped
    per-token at `jsd_token_clip` (default 0.05, their Qwen3-1.7B setting),
    THEN averaged. Defaults (`beta=0`, `top_k=None` i.e. full vocabulary,
    `temperature=1.1`) match every published OPSD launch script
    (scripts/run_opsd_*.sh) - NOT the paper-table-only guesses this file
    previously shipped with. `generated_ids` is accepted for interface
    parity with the sparsified path but unused when `top_k is None`.

    log_teacher must already be detached (cached teacher forward pass, no
    debiasing/projection - OPSD matches the raw teacher directly).
    """
    assert log_teacher.requires_grad is False
    log_teacher = log_teacher.detach()

    log_student_t = F.log_softmax(log_student / temperature, dim=-1)
    log_teacher_t = F.log_softmax(log_teacher / temperature, dim=-1)

    if top_k is not None:
        # Opt-in sparsification, OFF by default to match OPSD's real (full-
        # vocabulary) loss - kept only for anyone deliberately ablating it.
        from tropic.primitives import apply_sparse_mask, build_shared_topk_mask

        mask = build_shared_topk_mask(log_teacher_t, generated_ids, top_k)
        log_student_t = apply_sparse_mask(log_student_t, mask, default_mass)
        log_teacher_t = apply_sparse_mask(log_teacher_t, mask, default_mass)

    jsd_pointwise = generalized_jsd_pointwise(log_student_t, log_teacher_t, beta)  # [T, V]
    if jsd_token_clip is not None:
        jsd_pointwise = jsd_pointwise.clamp(max=jsd_token_clip)  # clip BEFORE the vocab sum
    jsd_per_token = jsd_pointwise.sum(dim=-1)  # [T]
    return jsd_per_token.mean()
