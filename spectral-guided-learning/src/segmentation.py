"""Step segmentation of long CoT responses with exact token-offset boundaries.

The paper doesn't specify a segmentation rule, so we split at sentence-ending punctuation
(or a closing brace/bracket, ending inline LaTeX) followed by whitespace and a capital letter.

Boundaries are found in character space and mapped to token offsets from a single tokenizer
pass over the whole response -- never tokenized piecewise, since BPE is context-dependent and
splitting per step would yield token ids the model was never pretrained on.
"""

import bisect
import re

_STEP_BOUNDARY = re.compile(r"([.?!\}\]])([\s\n]+)([A-Z])")


def split_into_step_texts(response: str) -> list[str]:
    """Split a response into step texts at sentence-level logical boundaries.

    Concatenating the returned pieces reproduces the original string exactly, so token
    offsets computed from cumulative prefixes stay aligned with the tokenized response.
    """
    if not response:
        return []

    pieces, cursor = [], 0
    for match in _STEP_BOUNDARY.finditer(response):
        boundary = match.start(3)  # cut before the capital letter, keep whitespace behind
        pieces.append(response[cursor:boundary])
        cursor = boundary
    if cursor < len(response):
        pieces.append(response[cursor:])
    return pieces


def record_step_spans(record: dict) -> list[tuple[int, int]]:
    """Absolute (start, end) token spans of a data_prep.py record's steps."""
    return [(step["token_start"], step["token_end"]) for step in record["steps"]]


def solution_step_start(step_texts: list[str]) -> int | None:
    """Index of the first step belonging to the final solution, for methods (Pru-CoT) that treat
    the CoT and solution as separate. Our source guarantees every response is
    "<think>\n{trajectory}\n</think>\n\n{attempt}" (data_prep.py); since the tag rarely lands on
    a step boundary, the step whose text *contains* "</think>" (and everything after) counts as
    solution. Returns None (whole-response fallback) when no tag is present.
    """
    for index, text in enumerate(step_texts):
        if "</think>" in text:
            return index
    return None


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[int]]:
    """Tokenize `text` once, returning its ids and each token's starting character index.

    Requires a fast (Rust-backed) tokenizer, which every `AutoTokenizer` used here is.
    """
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoding["input_ids"], [start for start, _ in encoding["offset_mapping"]]


def step_token_spans(
    response: str,
    token_starts: list[int],
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Map the response's step boundaries onto token indices, shifted by `offset`.

    A boundary landing inside a token assigns that whole token to the later step, since
    `bisect_left` returns the first token starting at or after the boundary. Zero-token
    steps (two boundaries falling within one token) are dropped so every span is non-empty.
    """
    cuts, position = [], 0
    for text in split_into_step_texts(response):
        cuts.append(position)
        position += len(text)

    bounds = [bisect.bisect_left(token_starts, cut) for cut in cuts] + [len(token_starts)]
    return [
        (offset + start, offset + end)
        for start, end in zip(bounds, bounds[1:])
        if end > start
    ]


def segment_response_token_spans(
    tokenizer,
    prompt_ids: list[int],
    response: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize a response and map each step to an absolute token span.

    Returns:
        (response_ids, step_spans) where spans are absolute positions in
        prompt_ids + response_ids.
    """
    response_ids, token_starts = encode_with_offsets(tokenizer, response)
    return response_ids, step_token_spans(response, token_starts, offset=len(prompt_ids))
