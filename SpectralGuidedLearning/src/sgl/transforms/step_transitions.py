"""Step structure for the next-step representation objective (L_trans).

Independent of the sentence-level segmentation in sgl.segment.sentence (which feeds spectral step
selection): here a step is a `\\n\\n`-delimited paragraph of the thinking trace, the same unit SGL
and step-entropy use. Steps shorter than `min_step_tokens` are folded into the step before them.

Per-token `step_id` (-1 outside the CoT), per-step `step_end` (absolute index of the token that
closes the step, i.e. the one carrying the `\\n\\n`) and `pair_src` (source step index i of every
kept transition i -> i+1) are what the training loss consumes.
"""

import bisect
import re

_PARAGRAPH_BREAK = re.compile(r"\n{2,}")  # "\n\n\n" is one boundary, not two
DEFAULT_MIN_STEP_TOKENS = 8


def cot_char_span(response: str) -> tuple[int, int]:
    """Character span of the thinking trace inside `response`.

    Covers the three response layouts sgl.data.prepare can emit: a full `<think>...</think>` block;
    only `</think>` (the prompt opened the block, R1-style); or no tags at all (the prompt already
    emitted an empty thinking block, e.g. Qwen3 with enable_thinking=False -- the trajectory and the
    final write-up are then plain prose, and both are stepped). Prompt tokens are never steps.
    """
    close = response.find("</think>")
    if close < 0:
        return 0, len(response)
    open_tag = response.find("<think>")
    start = open_tag + len("<think>") if 0 <= open_tag < close else 0
    return start, close


def split_cot_char_cuts(response: str) -> list[int]:
    """Character offsets where each step starts, plus the CoT end; every step keeps its
    trailing `\\n\\n`, so the boundary sits *after* the break and the break token closes the step."""
    start, end = cot_char_span(response)
    cot = response[start:end]
    if not cot.strip():
        return []
    cuts = [start]
    for match in _PARAGRAPH_BREAK.finditer(cot):
        if match.end() < len(cot):
            cuts.append(start + match.end())
    cuts.append(end)
    return cuts


def merge_short_steps(spans: list[tuple[int, int]], min_step_tokens: int) -> list[tuple[int, int]]:
    """Fold every span shorter than `min_step_tokens` into the span before it (the first span,
    lacking a predecessor, is folded into the one after it). Spans are contiguous, so a merge is
    just an extension of the previous end."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start < min_step_tokens or merged[-1][1] - merged[-1][0] < min_step_tokens):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def step_token_spans_from_cuts(
    cuts: list[int], token_starts: list[int], offset: int = 0
) -> list[tuple[int, int]]:
    """Map character cuts onto absolute token spans (a token straddling a cut goes to the later
    step, as in segmentation.step_token_spans). Empty spans are dropped."""
    if len(cuts) < 2:
        return []
    bounds = [bisect.bisect_left(token_starts, cut) for cut in cuts]
    return [(offset + a, offset + b) for a, b in zip(bounds, bounds[1:]) if b > a]


def build_transition_fields(
    response: str,
    token_starts: list[int],
    response_offset: int,
    sequence_length: int,
    min_step_tokens: int = DEFAULT_MIN_STEP_TOKENS,
) -> dict:
    """The transition fields for one record.

    Args:
        response: response text (sgl.data.prepare record["response"]).
        token_starts: starting char offset of every response token (encode_with_offsets).
        response_offset: absolute index of the first response token (response_token_span[0]).
        sequence_length: len(input_ids), to size `step_id`.
        min_step_tokens: steps shorter than this are merged; a transition whose target step is
            still shorter than this is dropped.

    Returns:
        {"step_id": int[T], "step_end": int[K], "num_steps": K, "pair_src": int[P]}
    """
    spans = step_token_spans_from_cuts(split_cot_char_cuts(response), token_starts, response_offset)
    spans = merge_short_steps(spans, min_step_tokens)

    step_id = [-1] * sequence_length
    for index, (start, end) in enumerate(spans):
        for position in range(start, min(end, sequence_length)):
            step_id[position] = index
    step_end = [end - 1 for _, end in spans]
    pair_src = [
        i for i in range(len(spans) - 1) if spans[i + 1][1] - spans[i + 1][0] >= min_step_tokens
    ]
    return {"step_id": step_id, "step_end": step_end, "num_steps": len(spans), "pair_src": pair_src}
