"""Phase 3: dual/triple-context forward passes on top of one shared model.

One model instance plays three roles that only differ in what is placed in
the context window (Section 3.1 of the paper):
    pi^S_t   student:    forward_student(...)    WITH grad
    pi^T_t   teacher:    forward_teacher(...)     no_grad (alpha=0 -> pi-bar^T_t == pi^T_t)
    pi^old_t checkpoint: forward_checkpoint(...)   no_grad, scores the CURRENT
                                                     (pre-update) parameters directly

Because teacher and student prefixes have different lengths (the teacher
prefix additionally contains the reference solution), each forward pass
slices its own logits at `prefix_len - 1 : -1` so that position `t` in the
returned tensor always corresponds to the same generated token `o_t` on both
sides - this offset bookkeeping is the single most error-prone part of the
whole construction (mirrors OPSD's `teacher_logits[:, teacher_prompt_len-1:-1,:]`).

CONTRACT: `forward_checkpoint` must be called for a given outer iteration
BEFORE any `optimizer.step()` happens in that same iteration (Algorithm 1's
"theta_k <- theta" is a snapshot taken once, before the inner-epoch updates).
The whole point of the fix below is that this no longer requires an actual
copy: at the moment `forward_checkpoint` runs, `self.model`'s parameters
*are* theta_k, because nothing has updated them yet this outer iteration -
we only need to read them, under no_grad, before the inner loop starts.

Three real bugs were fixed here across CPU and 24GB-GPU runs, in order:
  1. Device mismatch: prompt/generated-id tensors built on CPU (tokenizer
     output) were never moved to `model.device` before a CUDA forward pass -
     invisible on a CPU-only dev box, an immediate crash on GPU.
  2. `logits_to_keep`: OPSD's real reference solutions run to thousands of
     tokens, and a naive forward pass materializes logits for the FULL
     sequence ([1, L, V] with V~150k) even though we only ever read the last
     `gen_len` positions. Passing `logits_to_keep=gen_len+1` makes the model
     only project the last `gen_len+1` hidden states through the LM head,
     cutting that tensor from O(L*V) to O(gen_len*V).
  3. No copy at all for the checkpoint pass: an earlier version deep-copied
     the whole model into a temporary `old_model` every outer iteration to
     get pi^old. That's not just a memory cost (a second full model
     momentarily resident) but, at thousands of outer iterations, a real
     time cost (copying ~1.7B parameters, every step, for no reason) - since
     the live model already holds theta_k at the point this is called, we
     just score it directly (temporarily switching to `.eval()` so dropout,
     if any, doesn't make pi^old noisy, then switching back).

FIXED TEACHER (`fixed_teacher=True`, default off): OPSD's own real launch
scripts (scripts/run_opsd_*.sh, fetched live from github.com/siyan-zhao/OPSD
on 2026-09-02) all pass `--fixed_teacher`, i.e. their PUBLISHED numbers use a
teacher that is the frozen INITIAL policy (base model, LoRA adapters
disabled) for the whole run, not a teacher that evolves with the student.
Per their own README: "Our main results use the fixed teacher." Implemented
via PEFT's built-in `model.disable_adapter()` context manager - since a
freshly-initialized LoRA adapter's B matrix is zero, the adapter-disabled
forward pass is exactly the pretrained base model, needing no separate
snapshot or weight-swap machinery (contrast with PHF/PAINT's EMA/theta_0
teachers in an earlier version of this project, which genuinely needed one
since they don't map to "just turn the adapter off").
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ContextualPolicy:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, fixed_teacher: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.fixed_teacher = fixed_teacher
        if fixed_teacher and not hasattr(model, "disable_adapter"):
            raise ValueError(
                "fixed_teacher=True requires a PEFT model (disable_adapter() not found) - "
                "matches OPSD's own constraint (opsd_train.py: 'fixed_teacher=True requires use_peft=True')."
            )

    # -- shared forward + offset-slicing logic --------------------------------

    @staticmethod
    def _score_logits(model: PreTrainedModel, prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """Runs `model` on `prefix_ids + generated_ids` and returns RAW logits
        (pre-softmax) at exactly the `gen_len` positions that predicted each
        generated token (teacher-forcing shift), without ever materializing
        logits for the (potentially much longer) prefix. Split out from
        `_score` so TROPIC-G's debiasing (Eq. 11, `tropic.primitives.
        debias_logits`) can combine two RAW-logit forward passes (with-context
        minus alpha*context-only) BEFORE normalizing - subtracting two
        log_softmax outputs is NOT equivalent, since each carries its own
        normalization constant.
        """
        device = model.device
        prefix_ids = prefix_ids.to(device)
        generated_ids = generated_ids.to(device)
        gen_len = generated_ids.shape[0]
        full_ids = torch.cat([prefix_ids, generated_ids], dim=0).unsqueeze(0)
        out = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            logits_to_keep=gen_len + 1,
        )
        # out.logits: [1, gen_len+1, V] = positions [L-gen_len-1, L-1] inclusive.
        # We need the first gen_len of these (the last one predicts a token
        # past the end of the roll-out and is discarded).
        return out.logits[0, :gen_len, :]

    @classmethod
    def _score(cls, model: PreTrainedModel, prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """log_softmax over the vocabulary - thin wrapper over `_score_logits`
        for the (alpha=0, TROPIC-P) case where no debiasing combination is
        needed before normalizing."""
        return F.log_softmax(cls._score_logits(model, prefix_ids, generated_ids), dim=-1)

    @classmethod
    def _score_no_grad_eval(cls, model: PreTrainedModel, prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """Like `_score`, but temporarily forces `.eval()` (so dropout, if the
        model has any, doesn't make this read noisy) and always restores the
        model's previous training/eval mode afterwards - used for anything
        that scores the CURRENT live model without training it (teacher,
        checkpoint)."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return cls._score(model, prefix_ids, generated_ids).detach()
        finally:
            if was_training:
                model.train()

    @classmethod
    def _score_logits_no_grad_eval(cls, model: PreTrainedModel, prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """RAW-logit counterpart of `_score_no_grad_eval` - used by TROPIC-G's
        `forward_teacher_context_only` (Eq. 11 needs the pre-softmax logits,
        not log-probabilities - see `_score_logits`'s own docstring)."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return cls._score_logits(model, prefix_ids, generated_ids).detach()
        finally:
            if was_training:
                model.train()

    # -- the three forward passes ---------------------------------------------

    def forward_student(self, student_prompt_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """pi^S_t - WITH gradient."""
        return self._score(self.model, student_prompt_ids, generated_ids)

    def forward_teacher(self, teacher_prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """pi^T_t (alpha=0, so this already equals pi-bar^T_t - no extra
        forward pass over an empty-query context is computed, per TROPIC-P).
        When `fixed_teacher=True` (OPSD's own real recipe), the LoRA adapter
        is disabled for the duration of this forward pass so the teacher is
        scored under the frozen base (initial) policy instead of the current
        (updating) student weights."""
        if self.fixed_teacher:
            with self.model.disable_adapter():
                return self._score_no_grad_eval(self.model, teacher_prefix_ids, generated_ids)
        return self._score_no_grad_eval(self.model, teacher_prefix_ids, generated_ids)

    def forward_teacher_logits(self, teacher_prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """z_theta(.|x,o<t,r) - Eq. 11's first (with-context) term, as RAW
        LOGITS - same forward pass as `forward_teacher`, just before the
        final log_softmax, so TROPIC-G's `alpha>0` path can combine it with
        `forward_teacher_context_only`'s raw logits via `debias_logits`
        before normalizing once at the end."""
        if self.fixed_teacher:
            with self.model.disable_adapter():
                return self._score_logits_no_grad_eval(self.model, teacher_prefix_ids, generated_ids)
        return self._score_logits_no_grad_eval(self.model, teacher_prefix_ids, generated_ids)

    def forward_teacher_context_only(self, context_only_prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """z_theta(.|empty,o<t,r) - Eq. 11's second (context-only) term, used
        by TROPIC-**G** (alpha>0) only: TROPIC-P (alpha=0) never calls this at
        all (skip the extra forward pass entirely, per the module's own
        alpha=0 contract). `context_only_prefix_ids` is the SAME teacher
        template with the question replaced by an empty string (`tropic.data.
        build_teacher_context_only_prefix`) - the reference solution `r` is
        still present, only `x` is empty. Returns RAW LOGITS (not
        log-softmax) - the caller (`tropic.loss.tropic_p_loss` with
        `alpha>0`) must combine this with `forward_teacher`'s raw logits via
        `tropic.primitives.debias_logits` BEFORE normalizing."""
        if self.fixed_teacher:
            with self.model.disable_adapter():
                return self._score_logits_no_grad_eval(self.model, context_only_prefix_ids, generated_ids)
        return self._score_logits_no_grad_eval(self.model, context_only_prefix_ids, generated_ids)

    def forward_checkpoint(self, student_prompt_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """pi^old_t (Algorithm 1, theta_k <- theta). See the module-level
        CONTRACT note: must be called before this outer iteration's first
        `optimizer.step()`. Scores `self.model` directly - no copy."""
        return self._score_no_grad_eval(self.model, student_prompt_ids, generated_ids)

    def forward_all(
        self, student_prompt_ids: Tensor, teacher_prefix_ids: Tensor, generated_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convenience wrapper returning (log_student, log_teacher, log_old),
        with the shape-alignment assertion the plan calls out explicitly.
        Safe w.r.t. the CONTRACT above: none of the three forward passes
        update parameters by themselves (that only happens when the CALLER
        runs `loss.backward(); optimizer.step()`), so calling them in any
        order here is fine."""
        log_student = self.forward_student(student_prompt_ids, generated_ids)
        log_teacher = self.forward_teacher(teacher_prefix_ids, generated_ids)
        log_old = self.forward_checkpoint(student_prompt_ids, generated_ids)

        gen_len = generated_ids.shape[0]
        assert log_student.shape[0] == gen_len, (log_student.shape, gen_len)
        assert log_teacher.shape[0] == gen_len, (log_teacher.shape, gen_len)
        assert log_old.shape[0] == gen_len, (log_old.shape, gen_len)
        assert log_student.shape == log_teacher.shape == log_old.shape

        return log_student, log_teacher, log_old
