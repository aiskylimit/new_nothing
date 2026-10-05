"""Step segmentation: lossless rules (`"".join(split(text, mode)) == text`) plus the char -> token
span mapping every method shares.

    sentence   SGL: sentence-ending punctuation (or a closing brace/bracket) + whitespace + capital
    paragraph  Segment-Selective SFT fork default: every "\\n\\n" starts a segment
    cue        Segment-Selective SFT paper: "\\n\\n" followed by a backtracking cue ("Wait", ...)

P-ALIGN's sentence splitter is lossy and only used to build prefixes: see rules.palign_sentences.
"""
from sgl.segment.rules import cue_segments, palign_sentences, paragraph_segments, split_segments
from sgl.segment.sentence import (
    encode_with_offsets,
    record_step_spans,
    split_into_step_texts,
    step_token_spans,
)

SEGMENTERS = {
    "sentence": split_into_step_texts,
    "paragraph": paragraph_segments,
    "cue": cue_segments,
}


def split(text: str, mode: str = "sentence") -> list[str]:
    """Split `text` into steps with the named lossless rule."""
    if mode not in SEGMENTERS:
        raise ValueError(f"unknown segmenter {mode!r}; choose from {sorted(SEGMENTERS)}")
    if not text:
        return []
    return SEGMENTERS[mode](text)


__all__ = [
    "SEGMENTERS",
    "encode_with_offsets",
    "palign_sentences",
    "record_step_spans",
    "split",
    "split_segments",
    "step_token_spans",
]
