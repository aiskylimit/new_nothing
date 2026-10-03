"""RLSD (RLVR with Self-Distillation) baseline - a line-for-line port of
Algorithm 1 / Eq. 13-17 from "Self-Distilled RLVR" (arXiv:2604.03128,
Yang, Qin, Si et al., JD.COM/IIE-CAS).

FIDELITY NOTES (read before touching this file):

  - The paper's Eq. 15/16 (the "final objective") apply a PPO-style min/clip
    to `w_t * A`, which would be an importance-sampling-ratio-style clip on
    top of the credit weight. Algorithm 1's own pseudocode (line 17, the
    thing they actually run) instead does:
        A_hat_t = A * ((1 - lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w))
    i.e. a LINEAR INTERPOLATION between the plain GRPO uniform advantage
    (weight 1) and the reweighted advantage (weight clip(w_t)), scaled by a
    decaying `lambda` - "to avoid an abrupt transition at the start of
    training... gradually shifting to the uniform advantage" as lambda decays
    to 0 (Sec 4.2). We implement Algorithm 1's formula (the code, not the
    prose min/clip in Eq 15-16), since the user directed code over prose
    where the two disagree, and Algorithm 1 is unambiguous pseudocode with an
    explicit textual justification, whereas Eq 16 has no such gloss.
  - RLSD's per-token weight w_t = (P_T(y_t)/P_S(y_t))^sign(A) is evaluated
    ONLY at the actually-sampled token y_t (Eq 13-14: log P_T(y_t), log
    P_S(y_t) are scalars, not full-vocab distributions) - unlike OPSD/
    TROPIC-P/SDPO, which all match/diverge full [T,V] distributions. This is
    the single biggest structural difference: RLSD needs no full-vocab
    machinery at all, only `ContextualPolicy`'s existing full log_softmax
    output gathered at one index per position.
  - Delta_t is stop-gradiented (`sg(...)`, Eq 13) and the resulting A_hat_t is
    then ALSO fully detached before being used as a REINFORCE coefficient
    (Theorem 5 (i): "directional isolation" - r never enters the gradient
    DIRECTION, only a scalar magnitude multiplying grad log pi_theta(y_t)).
  - The teacher's privileged context r in this codebase's implementation is
    the GOLD FINAL ANSWER ONLY (`tropic.data.build_rlsd_teacher_prefix`),
    matching the paper's own description ("RLSD requires only the final
    ground-truth answer without any reasoning trace") - the paper itself
    doesn't publish an exact prompt template (different domain/dataset), so
    that builder is our own, clearly-labeled construction.
  - The group-relative advantage (Eq 1) is literally GRPO's; nothing new to
    port there beyond mean/std over one rollout group for the same question.
  - **TEACHER-FRESHNESS CORRECTION** (found by reading the REAL reference
    implementation, github.com/iie-ycx/RLSD, after this file's first version
    assumed a fully-live teacher from the paper's prose alone): the repo's
    `verl/trainer/opsd_config.py` dataclass DEFAULT (`freeze_teacher_model=
    False`, `rlsd_teacher_sync_interval=0`) is indeed fully live ("shares
    actor weights and updates every micro-batch") - but the actual PUBLISHED
    launch script that produces their results, `examples/visual_rl/
    rlsd_train.sh`, explicitly overrides this to `opsd.freeze_teacher_model=
    True opsd.rlsd_teacher_sync_interval=20`. The real mechanism (verified in
    `verl/workers/actor/dp_opsd_actor.py`'s `_get_teacher_module`/
    `_sync_teacher_from_actor`, and the tail of `update_policy_grpo_opsd`:
    `if global_step > 0 and global_step % sync_interval == 0:
    self._sync_teacher_from_actor()`) is a SEPARATE teacher module, holding a
    snapshot of the actor's weights, refreshed every `sync_interval` steps
    and otherwise frozen (`.eval()`, no grad) in between - "a middle ground
    between fully frozen (stale signal) and fully shared (unstable /
    distribution drift)" per the config's own docstring. The teacher starts
    as a frozen copy of the INITIAL weights (matches this file's own
    `PeriodicTeacherSync`, which syncs once at construction). Everything
    else this file computes with that teacher's log-probs (`rlsd_token_
    weight`, `rlsd_token_advantage`) is verified to EXACTLY match the real
    `_build_stgca_advantages` (same file) line-for-line - only WHICH
    weights produced `log_teacher_tok` needed correcting, not the math done
    with it.
"""
from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from tropic.chat_prompt import BOXED_INSTRUCTION, render_chat_prompt


def build_rlsd_teacher_prefix(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    gold_answer: str,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
    empty_think_suffix: str | None = None,
) -> torch.Tensor:
    """RLSD's teacher context r = the final ground-truth ANSWER ONLY, not a
    reasoning trace ("RLSD requires only the final ground-truth answer
    without any reasoning trace, making it the least demanding in terms of
    privileged information" - Sec 5.1). The paper itself (a multimodal-
    reasoning paper, different dataset/domain) does not publish an exact text
    template for this prompt, so the wording below is our own construction
    for this text-math setting - kept structurally parallel to this
    project's own `tropic.data.build_teacher_prefix` (same boxed-answer
    instruction, same single-user-message shape) so the only substantive
    difference between the OPSD/TROPIC-P teacher and the RLSD teacher is
    solution-vs-answer-only, exactly as the paper specifies."""
    content = (
        f"Problem: {question}\n\n"
        f"The final answer to this problem is: {gold_answer}\n\n"
        f"Using this knowledge, work out the reasoning that leads to this answer. "
        f"{BOXED_INSTRUCTION}"
    )
    return render_chat_prompt(tokenizer, content, enable_thinking, max_length, empty_think_suffix)


def group_relative_advantage(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Eq. 1 (GRPO, reused verbatim by RLSD): A^(i) = (R^(i) - mu_G) / sigma_G
    over one group of G rollouts sampled for the SAME question. `rewards`:
    [G] binary (or any scalar) verifier rewards. `eps` guards the division
    when every rollout in the group received an identical reward (sigma_G=0)
    - both source papers explicitly document this as a known GRPO
    degeneracy ("when all rollouts in a group receive the same reward, GRPO
    advantages collapse to zero"), not a bug to be hidden here."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def gather_token_logprob(log_probs: Tensor, token_ids: Tensor) -> Tensor:
    """log_probs: [T, V] full-vocab log_softmax (e.g. ContextualPolicy's
    `_score` output). token_ids: [T] the actually-generated tokens at each
    position. Returns [T]: log pi(y_t | ...) at just the sampled token - all
    RLSD ever needs from a full-vocab distribution (Eq. 13)."""
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def rlsd_token_weight(
    log_teacher_tok: Tensor, log_student_tok: Tensor, advantage: Tensor, epsilon_w: float
) -> Tensor:
    """Eq. 13-15. `log_teacher_tok`, `log_student_tok`: [T] per-token
    log-probs at the sampled tokens, BOTH must already be detached (Delta_t
    is stop-gradiented by construction - Sec 4.1: "the stop-gradient ensures
    that Delta_t serves purely as a weighting signal"). `advantage`: scalar
    A^(i) for this rollout (from `group_relative_advantage`).

    w_t = exp(sign(A) * Delta_t) = (P_T(y_t)/P_S(y_t))^sign(A), clipped to
    [1-eps_w, 1+eps_w] (Eq. 15) - always strictly positive, so the clipped
    weight can never flip the sign of A (Property 1, "direction anchoring").
    """
    assert not log_teacher_tok.requires_grad, "teacher log-prob must be detached (Sec 4.1 stop-gradient)"
    assert not log_student_tok.requires_grad, "student log-prob fed to Delta_t must be detached"
    delta_t = log_teacher_tok - log_student_tok
    sign_a = torch.sign(advantage)
    w_t = torch.exp(sign_a * delta_t)
    return torch.clamp(w_t, 1.0 - epsilon_w, 1.0 + epsilon_w)


def rlsd_token_advantage(advantage: Tensor, clipped_weight: Tensor, lam: float) -> Tensor:
    """Algorithm 1 line 17: A_hat_t = A * ((1-lambda) + lambda*clip(w_t,...)).
    At lambda=0 this collapses to plain GRPO (A_hat_t = A for every t,
    independent of the teacher) - the stated end-state of the paper's own
    lambda-decay schedule."""
    return advantage * ((1.0 - lam) + lam * clipped_weight)


def rlsd_lambda_schedule(step: int, lambda_init: float = 0.5, decay_steps: int = 50) -> float:
    """Sec 4.2 / Table (Implementation Details): "lambda... is initialized at
    0.5 and linearly decayed to 0 over the first 50 training steps." `step`
    is 0-indexed; held at 0.0 for any step >= decay_steps."""
    if decay_steps <= 0:
        return 0.0
    frac = max(0.0, 1.0 - step / decay_steps)
    return lambda_init * frac


def rlsd_loss(
    log_student_tok: Tensor,  # [T] WITH grad: log pi_theta(y_t | x, y<t)
    log_teacher_tok: Tensor,  # [T] teacher log-prob at y_t, detached
    advantage: Tensor,        # scalar A^(i), this rollout's group-relative advantage
    lam: float,
    epsilon_w: float,
) -> tuple[Tensor, Tensor]:
    """One rollout's RLSD loss (Eq. 13-17 + Algorithm 1 lines 12-22, single
    response). Returns (loss, w_t) - `loss` is a REINFORCE surrogate to
    MINIMIZE whose gradient reproduces nabla_theta J_RLSD (Eq. 38): A_hat_t
    is a scalar constant w.r.t. theta by construction (A comes from the
    external verifier reward; Delta_t/w_t are stop-gradiented), so
    `d/d(log_student_tok) [-(A_hat_t * log_student_tok).mean()] = -A_hat_t/T`
    matches `-nabla_theta J_RLSD` exactly. `w_t` is returned purely for
    diagnostics (mirrors `tropic_p_loss`'s `s` return), matching Fig 5(c)'s
    "clip ratio" logging.
    """
    assert not log_teacher_tok.requires_grad, "teacher log-prob must be detached"
    w_t = rlsd_token_weight(log_teacher_tok, log_student_tok.detach(), advantage, epsilon_w)
    a_hat_t = rlsd_token_advantage(advantage, w_t, lam).detach()
    loss = -(a_hat_t * log_student_tok).mean()
    return loss, w_t


class PeriodicTeacherSync:
    """Reproduces the REAL reference training script's teacher-freshness
    policy (see this module's TEACHER-FRESHNESS CORRECTION note above), not
    the library's inert `freeze_teacher_model=False` default: the teacher is
    a snapshot of the student's own (LoRA) weights, refreshed every
    `sync_interval` completed training steps and frozen in between -
    `sync_interval=0` recovers the fully-live behavior this file's first
    version used everywhere (`_sync_teacher_from_actor` never runs).

    `tropic.model.ContextualPolicy` only exposes two teacher modes (always
    live, or permanently frozen at init via PEFT's `disable_adapter()` -
    OPSD's own convention, see that module's docstring) - neither expresses
    "frozen but periodically refreshed", so this lives here rather than
    touching that shared file. Since the base model's weights never change
    (only the LoRA adapter trains), the snapshot only ever needs to hold the
    trainable tensors - no second full model copy.
    """

    def __init__(self, model, sync_interval: int):
        self.model = model
        self.sync_interval = sync_interval
        self._snapshot: dict[str, Tensor] | None = None
        if sync_interval > 0:
            self.sync()  # "the teacher starts as a frozen copy" of the init weights

    def _trainable_named_parameters(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def sync(self) -> None:
        """Refresh the snapshot from the model's CURRENT (live) weights."""
        self._snapshot = {n: p.detach().clone() for n, p in self._trainable_named_parameters()}

    def maybe_sync(self, step: int) -> bool:
        """Call once per completed training step (0-indexed), AFTER that
        step's `optimizer.step()` - mirrors the real code's post-update
        check (`if global_step > 0 and global_step % sync_interval == 0`).
        Returns whether a sync happened, for logging."""
        if self.sync_interval <= 0:
            return False
        if step > 0 and step % self.sync_interval == 0:
            self.sync()
            return True
        return False

    def forward_teacher(self, policy, teacher_prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """Scores `teacher_prefix_ids + generated_ids` under the last-synced
        snapshot (or live weights, if `sync_interval<=0`) rather than the
        model's current live parameters - temporarily swaps the snapshot
        onto `self.model`'s trainable tensors, scores via `policy.
        forward_teacher`, then restores the live values, so this call never
        perturbs the model being trained."""
        if self.sync_interval <= 0:
            return policy.forward_teacher(teacher_prefix_ids, generated_ids)

        params = self._trainable_named_parameters()
        live = {n: p.detach().clone() for n, p in params}
        with torch.no_grad():
            for n, p in params:
                p.data.copy_(self._snapshot[n])
        try:
            return policy.forward_teacher(teacher_prefix_ids, generated_ids)
        finally:
            with torch.no_grad():
                for n, p in params:
                    p.data.copy_(live[n])
