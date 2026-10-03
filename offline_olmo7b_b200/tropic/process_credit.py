"""TROPIC-PC (process-level credit), Sec 4.4 of TROPIC_PRM.pdf v6 - pure
segmentation/advantage/weighting primitives plus the SELF-VALUE (SV) source
(Eq. 10) and a unified teacher-anchor wrapper (Sec 4.2). Only SV is wired up
here by explicit request ("SV là phương pháp chính nhé, code cái này trước.
Các cách kia tôi chưa muốn thử") - the Outcome/SV+O/PRM sources described in
the paper are NOT implemented in this file yet.

Nothing here modifies tropic/model.py, tropic/ema_teacher.py, tropic/rlsd.py
or tropic/eval.py - AnchorTeacher below only WRAPS ContextualPolicy,
EMATeacherSync and PeriodicTeacherSync through their existing public methods.
"""
from __future__ import annotations

import re

import torch
from torch import Tensor

from tropic.primitives import normalize_log

# ---------------------------------------------------------------------------
# Segmentation (Sec 4.4): split a roll-out into reasoning steps at blank
# lines, merge short segments into their successor, cap the step count.
# ---------------------------------------------------------------------------


def _merge_short_segments(boundaries: list[int], min_len: int) -> list[int]:
    """Segments shorter than `min_len` tokens are merged into their
    successor (forward absorption) - a short FINAL segment has no successor,
    so it is folded into its predecessor instead."""
    if len(boundaries) <= 2:
        return boundaries
    merged = [boundaries[0]]
    i = 1
    while i < len(boundaries):
        seg_len = boundaries[i] - merged[-1]
        is_last = i == len(boundaries) - 1
        if seg_len < min_len and not is_last:
            i += 1  # extend this segment forward, absorbing boundaries[i]
            continue
        merged.append(boundaries[i])
        i += 1
    if len(merged) >= 3 and merged[-1] - merged[-2] < min_len:
        merged.pop(-2)  # fold a short final segment into its predecessor
    return merged


def _merge_to_max_steps(boundaries: list[int], max_steps: int) -> list[int]:
    """Repeatedly merges the two adjacent segments with the smallest
    combined length until at most `max_steps` segments remain (J <= J_max)."""
    boundaries = list(boundaries)
    while len(boundaries) - 1 > max_steps:
        seg_lens = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        if len(seg_lens) < 2:
            break
        combined = [seg_lens[i] + seg_lens[i + 1] for i in range(len(seg_lens) - 1)]
        drop_idx = combined.index(min(combined)) + 1
        boundaries.pop(drop_idx)
    return boundaries


_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")


def segment_steps(tokenizer, generated_ids: Tensor, min_len: int = 32, max_steps: int = 16) -> list[int]:
    """Sec 4.4: boundaries 0 = b_0 < b_1 < ... < b_J = len(generated_ids).

    Splits at blank lines in the DECODED text, maps each character split
    point back to a token index via binary search on `tokenizer.decode`'s
    output length (assumed monotonically non-decreasing in the number of
    tokens decoded - true for standard BPE tokenizers), then applies the
    merge rules above. Returns [0, n] (one step) when no blank line is found,
    and [0] when the roll-out is empty.
    """
    n = int(generated_ids.shape[0])
    if n == 0:
        return [0]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    split_chars = [m.start() for m in _BLANK_LINE_RE.finditer(full_text)]
    if not split_chars:
        return [0, n]

    def decoded_len(i: int) -> int:
        return len(tokenizer.decode(generated_ids[:i], skip_special_tokens=True))

    boundaries = {0, n}
    for char_pos in split_chars:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if decoded_len(mid) < char_pos:
                lo = mid + 1
            else:
                hi = mid
        if 0 < lo < n:
            boundaries.add(lo)

    result = sorted(boundaries)
    result = _merge_short_segments(result, min_len)
    result = _merge_to_max_steps(result, max_steps)
    return result


def select_evaluation_boundaries(num_steps: int, max_boundaries: int) -> list[int]:
    """Algorithm 1 line 7: "evaluate V at <= m boundaries per roll-out".
    Returns <= max_boundaries indices into {0, ..., num_steps}, evenly
    spaced, ALWAYS including both endpoints 0 and num_steps."""
    total = num_steps + 1
    if max_boundaries >= total:
        return list(range(total))
    idx = torch.linspace(0, num_steps, steps=max(max_boundaries, 2)).round().long()
    return sorted(set(idx.tolist()) | {0, num_steps})


def expand_sparse_values(evaluated_indices: list[int], evaluated_values: Tensor, num_steps: int) -> Tensor:
    """"Steps between evaluated boundaries share the increment uniformly"
    (Sec 4.4): linearly interpolates values at every UNEVALUATED boundary
    between the two nearest evaluated ones, so each of the `span` steps in
    between gets an equal increment (v1-v0)/span. Returns values_full:
    [num_steps+1]."""
    values_full = torch.zeros(num_steps + 1, dtype=evaluated_values.dtype)
    for k in range(len(evaluated_indices) - 1):
        i0, i1 = evaluated_indices[k], evaluated_indices[k + 1]
        v0, v1 = evaluated_values[k], evaluated_values[k + 1]
        span = i1 - i0
        for offset in range(span + 1):
            values_full[i0 + offset] = v0 + (v1 - v0) * (offset / span)
    return values_full


# ---------------------------------------------------------------------------
# Eq. 11 - the process advantage, and Algorithm 1's optional first-error
# emphasis (Sec 4.4).
# ---------------------------------------------------------------------------


def batch_sigma_hat(increments: Tensor, sigma_min: float) -> Tensor:
    """sigma_hat = std of value increments in the CURRENT (micro-)batch,
    floored at sigma_min (Eq. 11). Caller collects every increment across
    every rollout in the current training step and passes the flattened
    tensor here ONCE, so every rollout's `progress_advantage` call in that
    step shares the same scale."""
    if increments.numel() < 2:
        return torch.as_tensor(sigma_min, dtype=torch.get_default_dtype())
    return torch.clamp(increments.std(unbiased=False), min=sigma_min)


def progress_advantage(values: Tensor, sigma_hat) -> Tensor:
    """Eq. 11: Ahat_j = clip((V(o<=b_j) - V(o<=b_{j-1})) / sigma_hat, -1, 1).

    `values`: [J+1] (V at every boundary b_0..b_J). Returns Ahat: [J].
    """
    increments = values[1:] - values[:-1]
    return torch.clamp(increments / sigma_hat, -1.0, 1.0)


def token_step_index(boundaries: list[int]) -> Tensor:
    """boundaries: [b_0=0, ..., b_J=T]. Returns step_of_token: [T] int64,
    where step_of_token[t] in {0, ..., J-1} is the 0-indexed step containing
    token position t - used to broadcast per-step Ahat/weights to tokens."""
    parts = []
    for j in range(len(boundaries) - 1):
        length = boundaries[j + 1] - boundaries[j]
        parts.append(torch.full((length,), j, dtype=torch.long))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)


def tilt_per_token(advantages: Tensor, step_of_token: Tensor, kappa: float) -> Tensor:
    """kappa * Ahat_j(t), broadcast from per-step [J] to per-token [T] -
    exactly the quantity `tropic.loss_pc.tropic_pc_loss`'s `tilt` argument
    expects (Eq. 12)."""
    if advantages.numel() == 0 or step_of_token.numel() == 0:
        return torch.zeros_like(step_of_token, dtype=torch.get_default_dtype())
    return kappa * advantages[step_of_token]


def first_error_weights(advantages: Tensor, omega: float, step_of_token: Tensor) -> Tensor:
    """Optional first-error emphasis (Sec 4.4 / Algorithm 1 line 8): let
    j* = argmin_j Ahat_j. If Ahat_j* < 0, the per-token loss on step j* is
    weighted by (1+omega) and all weights renormalized to mean one.
    Returns w: [T], mean(w) == 1 (or all-ones if there are no steps)."""
    J = advantages.shape[0]
    if J == 0 or step_of_token.numel() == 0:
        return torch.ones_like(step_of_token, dtype=torch.get_default_dtype())
    j_star = int(torch.argmin(advantages).item())
    weights_per_step = torch.ones(J, dtype=advantages.dtype)
    if advantages[j_star].item() < 0:
        weights_per_step[j_star] = 1.0 + omega
    w_t = weights_per_step[step_of_token]
    return w_t / w_t.mean()


# ---------------------------------------------------------------------------
# Sec 4.2 - a unified interface over the three teacher anchors this codebase
# already implements (frozen / EMA / periodic sync), WITHOUT modifying any
# of them.
# ---------------------------------------------------------------------------


class AnchorTeacher:
    """Dispatches to whichever anchor implementation is already in this
    codebase, all through their existing PUBLIC methods:

      - "frozen" (tau=1, OPSD's real recipe): `policy` must have been built
        with `fixed_teacher=True` (tropic.model.ContextualPolicy); scoring
        goes straight through `policy.forward_teacher` (disable_adapter()).
      - "ema" (tau in (0,1)): wraps `tropic.ema_teacher.EMATeacherSync`.
      - "sync" (hard sync every M steps): wraps
        `tropic.rlsd.PeriodicTeacherSync`.

    `score_log_probs` always returns log_softmax'd log-probabilities
    (matching `ContextualPolicy.forward_teacher`'s convention), so
    `compute_self_value` below never needs to know which anchor is active.
    """

    def __init__(self, mode: str, policy, model=None, ema_decay: float | None = None, sync_interval: int | None = None):
        if mode not in ("frozen", "ema", "sync"):
            raise ValueError(f"unknown anchor mode {mode!r}, expected 'frozen', 'ema' or 'sync'")
        self.mode = mode
        self.policy = policy
        self._impl = None
        if mode == "frozen":
            if not getattr(policy, "fixed_teacher", False):
                raise ValueError("anchor='frozen' requires ContextualPolicy(..., fixed_teacher=True)")
        elif mode == "ema":
            from tropic.ema_teacher import EMATeacherSync

            if ema_decay is None:
                raise ValueError("anchor='ema' requires ema_decay")
            self._impl = EMATeacherSync(model, ema_decay)
        else:
            from tropic.rlsd import PeriodicTeacherSync

            if sync_interval is None:
                raise ValueError("anchor='sync' requires sync_interval")
            self._impl = PeriodicTeacherSync(model, sync_interval)

    def update(self, step: int | None = None) -> None:
        """Call once per completed training step, AFTER optimizer.step()
        (same contract as EMATeacherSync.update/PeriodicTeacherSync.maybe_sync)."""
        if self.mode == "ema":
            self._impl.update()
        elif self.mode == "sync":
            if step is None:
                raise ValueError("anchor='sync' requires `step` on update()")
            self._impl.maybe_sync(step)
        # frozen: no-op - permanently frozen at init, matches ContextualPolicy's own contract.

    def score_log_probs(self, prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """log pi_phi_k(generated_ids | prefix_ids) under this anchor's
        CURRENT weights - detached, no_grad (all three underlying
        implementations already guarantee this)."""
        if self.mode == "frozen":
            return self.policy.forward_teacher(prefix_ids, generated_ids)
        if self.mode == "sync":
            return self._impl.forward_teacher(self.policy, prefix_ids, generated_ids)
        # ema: EMATeacherSync only exposes a RAW-LOGITS forward (shared with
        # the Eq. 9 debiasing path) - normalize it ourselves.
        raw_logits = self._impl.forward_teacher_logits(self.policy, prefix_ids, generated_ids)
        return normalize_log(raw_logits)

    def score_logits(self, prefix_ids: Tensor, generated_ids: Tensor) -> Tensor:
        """RAW logits (pre-softmax) under this anchor - Eq. 9/11's debiasing
        needs to combine two RAW-logit forward passes (with-context minus
        alpha*context-only) BEFORE normalizing (see tropic.model.
        ContextualPolicy.forward_teacher_logits's own docstring for why).
        Only "frozen" and "ema" expose a public raw-logits method on the
        wrapped class (ContextualPolicy.forward_teacher_logits /
        EMATeacherSync.forward_teacher_logits respectively); "sync"
        (tropic.rlsd.PeriodicTeacherSync) only exposes a log_softmax'd
        forward_teacher, so TROPIC-G-style debiasing (alpha>0) under a
        periodically-synced anchor is not available without modifying that
        file - use alpha=0 with anchor='sync', or a different anchor mode."""
        if self.mode == "frozen":
            return self.policy.forward_teacher_logits(prefix_ids, generated_ids)
        if self.mode == "ema":
            return self._impl.forward_teacher_logits(self.policy, prefix_ids, generated_ids)
        raise NotImplementedError(
            "anchor='sync' (tropic.rlsd.PeriodicTeacherSync) has no raw-logits forward pass, "
            "so TROPIC-G-style debiasing (alpha>0) is not supported with this anchor - "
            "pass alpha=0.0, or use anchor='frozen'/'ema' instead."
        )


# ---------------------------------------------------------------------------
# Eq. 10 - the self-value (SV) source. THE ONLY VALUE SOURCE IMPLEMENTED SO
# FAR, by explicit request; Outcome/SV+O/PRM are not built here yet.
# ---------------------------------------------------------------------------


def compute_self_value(
    anchor: AnchorTeacher,
    student_prompt_ids: Tensor,
    generated_ids: Tensor,
    boundary: int,
    suffix_ids: Tensor,
    answer_ids: Tensor,
) -> Tensor:
    """Eq. 10: V^sv(o<=boundary) = exp( mean_i log pi_phi_k(a*_i | x, o<=boundary, u, a*_<i) ).

    `student_prompt_ids` (x) is the STUDENT prompt (tropic.data.
    build_student_prompt) and must NEVER contain the reference solution r -
    excluding r is essential since a* appears in r (Eq. 10's own note).
    This holds by construction here: the scored prefix below is built ONLY
    from `student_prompt_ids` + the roll-out prefix + the answer-forcing
    suffix `u` - `reference_solution`/`r` never enters this function at all
    (see tests/test_process_credit.py's leakage assertion, mirroring
    tropic.data.debug_print_examples's own check).

    Scoring runs under the ANCHOR's weights (phi_k), not the live student
    theta - "prevents the student from inflating its own value estimate"
    (Eq. 10's own note) - via `anchor.score_log_probs`, which is exactly
    `ContextualPolicy._score_no_grad_eval`'s no_grad/eval-mode contract
    reused through whichever of {frozen, ema, sync} is active.

    Returns a scalar Tensor in [0, 1] (it is exp of a mean of <=0 log-probs,
    so this holds by construction, not by clamping).

    All of `student_prompt_ids`/`generated_ids`/`suffix_ids`/`answer_ids` are
    moved onto `generated_ids`'s device before concatenating - callers don't
    need to pre-align devices themselves (mirrors `ContextualPolicy.
    _score_logits`'s own `.to(device)` calls, since a training script's
    tokenized prompts/suffixes are typically built on CPU while a vLLM-
    generated `generated_ids` already lives on the training GPU).
    """
    with torch.no_grad():
        device = generated_ids.device
        prefix_ids = torch.cat([
            student_prompt_ids.to(device), generated_ids[:boundary], suffix_ids.to(device),
        ])
        answer_ids = answer_ids.to(device)
        log_probs = anchor.score_log_probs(prefix_ids, answer_ids)  # [|a*|, V]
        gold_log_probs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)  # [|a*|]
        return gold_log_probs.mean().exp()


def compute_self_value_at_boundaries(
    anchor: AnchorTeacher,
    student_prompt_ids: Tensor,
    generated_ids: Tensor,
    boundaries: list[int],
    suffix_ids: Tensor,
    answer_ids: Tensor,
) -> Tensor:
    """Convenience: `compute_self_value` at every boundary in `boundaries`,
    stacked into one [len(boundaries)] tensor."""
    return torch.stack(
        [compute_self_value(anchor, student_prompt_ids, generated_ids, b, suffix_ids, answer_ids) for b in boundaries]
    )


# =============================================================================
# TROPIC_Proposal_v8, Section 4.4-4.7: LOG-SPACE process value, per-roll-out
# normalization with a dead zone, a
# */\boxed{}-extracted answer (no `Answer`
# column), and a segmentation FALLBACK LADDER. Added alongside (not
# replacing) the v6/v7 functions above - `compute_self_value`/
# `progress_advantage`/`segment_steps`/`AnchorTeacher` are all UNCHANGED and
# still used by scripts/run_tropic_pc_4b.py; nothing below alters their
# behavior. v8's own Section 4.4 identifies exponentiating the value
# (v6/v7's V^sv in probability space, `compute_self_value` above) and
# normalizing increments at the BATCH level (`progress_advantage` above) as
# the two most likely reasons the first process-credit run behaved like
# noise - both are fixed in the functions below, under new names, so v6/v7's
# own code path is never silently changed underneath it.
# =============================================================================


def extract_final_answer(reference_solution: str) -> str | None:
    """a* = a*(r) (v8's own notation, Section 3.1): the content of the LAST
    \\boxed{} in the reference solution r - a deterministic substring of the
    privileged context, NOT the dataset's separate `Answer` column (v8 is
    "verifier-free by construction": Section 1's changelog, "The final
    answer is read from the last \\boxed{} of r, not from a separate
    label"). Reuses `tropic.eval.extract_boxed_answer`'s exact brace-balanced
    parser UNCHANGED rather than reimplementing it. Returns None if `r` has
    no \\boxed{} at all - v8's own Section 3.1: "examples without one are
    dropped (their fraction is reported)"; the caller is responsible for
    dropping/reporting such examples, this function only signals the absence."""
    from tropic.eval import extract_boxed_answer

    return extract_boxed_answer(reference_solution)


def compute_logvalue(
    anchor: AnchorTeacher,
    student_prompt_ids: Tensor,
    generated_ids: Tensor,
    boundary: int,
    suffix_ids: Tensor,
    answer_ids: Tensor,
) -> Tensor:
    """Eq. 10 (v8): v(o<=boundary) = (1/|a*|) * sum_i log pi_phi_k(a*_i | x,
    o<=boundary, u, a*_<i) - the MEAN LOG-LIKELIHOOD itself, in nats, NEVER
    exponentiated. Section 4.4: "the value is kept in nats and never
    exponentiated... on competition problems the corresponding probability
    is often below 1e-3, where differences between prefixes are numerically
    negligible while the same differences in log space are of order one."

    Same leakage/anchor/device contract as `compute_self_value` above (see
    its own docstring): `student_prompt_ids` never contains the reference
    solution, and scoring runs under the anchor's weights (phi_k), not the
    live student theta. The only difference from `compute_self_value` is the
    missing final `.exp()`.
    """
    with torch.no_grad():
        device = generated_ids.device
        prefix_ids = torch.cat([
            student_prompt_ids.to(device), generated_ids[:boundary], suffix_ids.to(device),
        ])
        answer_ids = answer_ids.to(device)
        log_probs = anchor.score_log_probs(prefix_ids, answer_ids)  # [|a*|, V]
        gold_log_probs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)  # [|a*|]
        return gold_log_probs.mean()


def compute_logvalue_at_boundaries(
    anchor: AnchorTeacher,
    student_prompt_ids: Tensor,
    generated_ids: Tensor,
    boundaries: list[int],
    suffix_ids: Tensor,
    answer_ids: Tensor,
) -> Tensor:
    """Convenience: `compute_logvalue` at every boundary in `boundaries`,
    stacked into one [len(boundaries)] tensor (nats, unbounded sign)."""
    return torch.stack(
        [compute_logvalue(anchor, student_prompt_ids, generated_ids, b, suffix_ids, answer_ids) for b in boundaries]
    )


def is_degenerate_rollout(deltas: Tensor, threshold: float) -> bool:
    """Section 4.4: "Roll-outs with max_j delta_j - min_j delta_j below a
    threshold receive no credit at all" - a ROLL-OUT-LEVEL guard, decided
    BEFORE any per-step credit is computed, distinct from `progress_v8`'s
    own per-step dead zone (`delta_min`). Stops a roll-out whose value never
    really moves from having pure numerical noise rescaled into saturated
    +-1 credits by per-roll-out normalization."""
    if deltas.numel() == 0:
        return True
    return (deltas.max() - deltas.min()).item() < threshold


def progress_v8(deltas: Tensor, sigma_min: float, delta_min: float) -> Tensor:
    """Eq. 11 (v8): Ahat_j = clip((delta_j - delta_bar)/sigma_hat, -1, 1) *
    1[|delta_j - delta_bar| > delta_min], where delta_bar/sigma_hat are the
    mean/std of THIS ROLL-OUT'S OWN {delta_j} - never a batch-level
    statistic (Section 4.4: "two details matter and were wrong in v7...
    normalizing per roll-out removes the problem-difficulty offset, which a
    batch-level statistic cannot do when easy and hard problems are mixed").
    `sigma_hat` is floored at the ABSOLUTE nat-space constant `sigma_min`
    (not a fraction of anything). The dead zone `delta_min` zeroes out a
    step's credit when it is not meaningfully different from this roll-out's
    OWN average step, guarding against numerical noise being rescaled into
    a saturated +-1 credit.

    `deltas`: [J] = v_j - v_{j-1} for j=1..J, from a SINGLE roll-out (call
    `is_degenerate_rollout` on the same `deltas` first - Section 4.4's
    roll-out-level zero-credit rule is NOT applied inside this function,
    since that decision is made once per roll-out, not derivable from
    `deltas` alone once already zeroed by the caller). Returns Ahat: [J].
    """
    if deltas.numel() == 0:
        return deltas
    delta_bar = deltas.mean()
    sigma_hat = deltas.std(unbiased=False).clamp(min=sigma_min)
    raw = (deltas - delta_bar) / sigma_hat
    active = (deltas - delta_bar).abs() > delta_min
    return torch.clamp(raw, -1.0, 1.0) * active.to(raw.dtype)


# ---------------------------------------------------------------------------
# Segmentation fallback ladder (Section 4.4 + Section 6.3 diagnostic #3:
# "if most roll-outs give J=1, enable the fallback segmentation before
# anything else"). blank-line -> single newline -> fixed ~chunk_size-token
# chunks (each nudged to the nearest sentence/line break). Self-contained
# (does not call the v6/v7 `segment_steps`/`_merge_short_segments` above by
# accident of shared state - `_merge_to_max_steps` is reused since it is a
# pure list operation with no coupling to how boundaries were found).
# ---------------------------------------------------------------------------

_SINGLE_NEWLINE_RE = re.compile(r"\n")
_SENTENCE_OR_LINE_BREAK_RE = re.compile(r"[.!?]\s|\n")


def _char_positions_to_token_boundaries(tokenizer, generated_ids: Tensor, char_positions: list[int]) -> set[int]:
    n = int(generated_ids.shape[0])

    def decoded_len(i: int) -> int:
        return len(tokenizer.decode(generated_ids[:i], skip_special_tokens=True))

    boundaries = {0, n}
    for char_pos in char_positions:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if decoded_len(mid) < char_pos:
                lo = mid + 1
            else:
                hi = mid
        if 0 < lo < n:
            boundaries.add(lo)
    return boundaries


def _segment_by_pattern(tokenizer, generated_ids: Tensor, pattern: re.Pattern, min_len: int, max_steps: int) -> list[int]:
    n = int(generated_ids.shape[0])
    if n == 0:
        return [0]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    split_chars = [m.start() for m in pattern.finditer(full_text)]
    if not split_chars:
        return [0, n]
    boundaries = sorted(_char_positions_to_token_boundaries(tokenizer, generated_ids, split_chars))
    boundaries = _merge_short_segments(boundaries, min_len)
    return _merge_to_max_steps(boundaries, max_steps)


def _segment_by_chunks(tokenizer, generated_ids: Tensor, chunk_size: int, max_steps: int) -> list[int]:
    """Last, always-available rung of the ladder: boundaries every
    ~chunk_size tokens, each nudged to the nearest sentence/line break
    (within a generous character window) so a chunk boundary does not
    routinely land mid-word/mid-clause - falls back to the raw token
    boundary when no break is found nearby."""
    n = int(generated_ids.shape[0])
    if n == 0:
        return [0]
    if n <= chunk_size:
        return [0, n]

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    break_positions = [m.end() for m in _SENTENCE_OR_LINE_BREAK_RE.finditer(full_text)]

    def decoded_len(i: int) -> int:
        return len(tokenizer.decode(generated_ids[:i], skip_special_tokens=True))

    raw_token_boundaries = list(range(0, n, chunk_size))
    if raw_token_boundaries[-1] != n:
        raw_token_boundaries.append(n)

    # ~4 chars/token is a rough, deliberately generous heuristic just to
    # size the nudge-search window - it never needs to be exact, since a
    # missed nudge just falls back to the raw token boundary.
    window_chars = max(8, chunk_size * 2)

    char_targets = []
    for raw in raw_token_boundaries[1:-1]:
        char_pos = decoded_len(raw)
        candidates = [p for p in break_positions if abs(p - char_pos) <= window_chars]
        char_targets.append(min(candidates, key=lambda p: abs(p - char_pos)) if candidates else char_pos)

    boundaries = sorted(_char_positions_to_token_boundaries(tokenizer, generated_ids, char_targets))
    return _merge_to_max_steps(boundaries, max_steps)


def segment_steps_v8(
    tokenizer,
    generated_ids: Tensor,
    min_len: int = 32,
    max_steps: int = 16,
    min_steps: int = 4,
    chunk_size: int = 96,
) -> tuple[list[int], str]:
    """Segmentation with the v8 fallback ladder: blank line -> single
    newline -> fixed ~chunk_size-token chunks. Returns (boundaries,
    fallback_level), fallback_level in {"blank_line", "newline", "chunk"}
    recording which rung was actually used - the segments diagnostic
    (scripts/diagnose_signal.py's `segments` subcommand) logs this
    distribution directly.

    Falls through to the next rung only when the current one does not reach
    `min_steps` steps AND the roll-out is long enough to plausibly support
    more (n >= min_steps * min_len) - a genuinely short roll-out is allowed
    to have fewer than `min_steps` steps at every rung; there is no point
    forcing more boundaries than the text can bear.
    """
    n = int(generated_ids.shape[0])
    if n == 0:
        return [0], "blank_line"

    def long_enough_to_expect_more_steps(n_tokens: int) -> bool:
        return n_tokens >= min_steps * min_len

    boundaries = _segment_by_pattern(tokenizer, generated_ids, _BLANK_LINE_RE, min_len, max_steps)
    if len(boundaries) - 1 >= min_steps or not long_enough_to_expect_more_steps(n):
        return boundaries, "blank_line"

    boundaries_nl = _segment_by_pattern(tokenizer, generated_ids, _SINGLE_NEWLINE_RE, min_len, max_steps)
    if len(boundaries_nl) - 1 >= min_steps or not long_enough_to_expect_more_steps(n):
        return boundaries_nl, "newline"

    boundaries_chunk = _segment_by_chunks(tokenizer, generated_ids, chunk_size, max_steps)
    return boundaries_chunk, "chunk"
