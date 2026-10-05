"""SDPO (Self-Distillation Policy Optimization) baseline - a port of the
distillation loss from "Reinforcement Learning via Self-Distillation"
(arXiv:2601.20802, Hubotter, Lubeck, Behric, Baumann et al., ETH Zurich),
verified against the REAL reference implementation at
https://github.com/lasgroup/SDPO (verl/trainer/ppo/core_algos.py's
`compute_self_distillation_loss`, verl/trainer/config/sdpo.yaml,
run_local_sdpo.sh) - not just the paper text.

FIDELITY NOTES (an earlier version of this file got one thing wrong - see
the correction below, kept here so nobody re-introduces it):

  - Self-teacher reprompt (Table 2 in the paper, unaffected by the code
    audit below): specialized to our math/RLVR-without-rich-feedback
    setting - no `environment_output` paragraph (our verifier only returns
    pass/fail, matching Sec 3's "Learning without Rich Environment Feedback",
    the closest match to a plain math-RLVR environment), and the student's
    own original attempt is deliberately NOT included (Table 6: including it
    "reduces entropy... thereby reducing exploration"). See
    `build_sdpo_self_teacher_prefix`.
  - **CORRECTION**: an earlier version of this file read Appendix A.2's
    "regularized self-teacher" (EMA of student weights, OR an explicit
    trust-region log-linear interpolation with a FROZEN INITIAL teacher,
    Eq. 10) as something the real training loop needs, and implemented a
    second `disable_adapter()` forward pass to build a frozen reference
    teacher, log-linearly blended with the live teacher via a `alpha`
    parameter. Reading the REAL code (`compute_self_distillation_loss`)
    proves this was a misreading: the `self_distillation.alpha` config
    (default 0.5 in run_local_sdpo.sh, the paper's own reference launch
    script) is NOT a frozen-teacher interpolation weight at all - it is the
    **generalized Jensen-Shannon mixture weight between STUDENT and
    TEACHER**, structurally IDENTICAL to OPSD's own `beta` parameter
    (`tropic.baselines.generalized_jsd_pointwise`, reused verbatim below):
        alpha=0.0 -> KL(teacher || student)   (special-cased, not the limit)
        alpha=1.0 -> KL(student || teacher)   (special-cased, not the limit;
                                                matches the paper's literal
                                                Eq. 1 forward-KL-through-
                                                student formula)
        else      -> alpha*KL(teacher||M) + (1-alpha)*KL(student||M),
                     M = (1-alpha)*student + alpha*teacher  (mixture)
    The real reference script's default alpha=0.5 is exactly the SYMMETRIC
    Jensen-Shannon divergence (Sec 2.3's stability recommendation). There is
    NO second/frozen-teacher forward pass anywhere in the real loss function
    - the self-teacher is a SINGLE live forward pass (Eq. 1's literal
    `stopgrad(pi_theta(.|x,f,y<t))`), full stop. The
    `sdpo_regularized_teacher_logprob` function and the `disable_adapter()`
    reference-teacher pass this file/its callers previously had are REMOVED
    - they implemented a mechanism the real algorithm does not use.
  - Approximate logit distillation (Eq. 11 / real code's `full_logit_
    distillation` + `distillation_topk` path): top-K tokens (by STUDENT
    mass) plus a tail bucket carrying the remaining probability - this is
    exactly what `tropic.primitives.build_shared_topk_mask` + `apply_sparse_
    mask` already implement (built for TROPIC-P's Remark 4 sparsification),
    reused verbatim here with `log_ref=log_student` ("the top-K is with
    respect to student" - paper text; the real code's topk selection is
    likewise student-side, `student_topk_log_probs`/`teacher_topk_log_probs`
    fed in externally). `distillation_topk=100` in the real reference script
    matches this file's own default.
"""
from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from tropic.baselines import generalized_jsd_pointwise
from tropic.chat_prompt import BOXED_INSTRUCTION, render_chat_prompt
from tropic.primitives import apply_sparse_mask, build_shared_topk_mask


def build_sdpo_self_teacher_prefix(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    successful_previous_rollout: str | None,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """SDPO's self-teacher reprompt (Table 2), specialized to our math/RLVR-
    without-rich-feedback setting: no `environment_output` paragraph (our
    verifier only returns pass/fail, not runtime errors/judge text -
    matching Sec 3's "Learning without Rich Environment Feedback" setting,
    the closest match to a plain math-RLVR environment), and the student's
    own original attempt is deliberately NOT included (Table 6: including it
    "biases the teacher towards the student's attempt" and "reduces
    entropy... thereby reducing exploration"). `successful_previous_rollout`
    is the DECODED TEXT of a correct rollout sampled by the student itself
    earlier in the same group (never an external/expert solution - "these
    sample solutions are always generated by the student, as in GRPO"); pass
    None when no rollout in the group graded correct, which renders the
    "Correct solution:" paragraph verbatim per Table 2's own skip-if-absent
    rule."""
    solution_block = (
        f"Correct solution:\n{successful_previous_rollout}\n\n" if successful_previous_rollout else ""
    )
    content = (
        f"Problem: {question}\n\n"
        f"{solution_block}"
        f"Correctly solve the original question. {BOXED_INSTRUCTION}"
    )
    return render_chat_prompt(tokenizer, content, enable_thinking, max_length, empty_think_suffix)


def sdpo_loss(
    log_student: Tensor,       # [T, V] WITH grad
    log_teacher: Tensor,       # [T, V] detached (stopgrad self-teacher target, LIVE weights, single pass)
    generated_ids: Tensor,     # [T]
    top_k: int,
    default_mass: float,
    alpha: float = 0.5,
) -> Tensor:
    """The real `compute_self_distillation_loss`'s `full_logit_distillation`
    path (top-K + tail variant), verified against
    verl/trainer/ppo/core_algos.py: `alpha` is the generalized Jensen-Shannon
    mixture weight between student and teacher (see module docstring for the
    exact special-cased formula) - `generalized_jsd_pointwise` (built for
    OPSD's own `beta`) implements the IDENTICAL formula, reused verbatim
    here with `beta=alpha`. Default `alpha=0.5` matches the real reference
    script's (run_local_sdpo.sh) default exactly - the symmetric JSD Sec 2.3
    recommends for stability. `alpha=1.0` recovers the paper's literal Eq. 1
    (KL(student || stopgrad(teacher))); `alpha=0.0` gives the reverse
    direction (KL(teacher || student))."""
    assert not log_teacher.requires_grad, "self-teacher must be detached (Eq. 1's stopgrad)"
    log_teacher = log_teacher.detach()

    mask = build_shared_topk_mask(log_student.detach(), generated_ids, top_k)
    log_student_sp = apply_sparse_mask(log_student, mask, default_mass)
    log_teacher_sp = apply_sparse_mask(log_teacher, mask, default_mass)

    per_token = generalized_jsd_pointwise(log_student_sp, log_teacher_sp, beta=alpha).sum(dim=-1)
    return per_token.mean()
