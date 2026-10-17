"""Segmentation rules of Segment-Selective SFT and P-ALIGN, ported verbatim.

`split_segments` is SegmentSelectiveSFT/segment_utils.py: the delimiter starts the segment that
follows it, `"".join(segments) == text`, and empty segments are merged away (a "\\n\\n" split leaves
empty strings at a leading "\\n\\n" or 4+ newlines, which would be zero-token segments).

`palign_sentences` is P-ALIGN/src/binary_search.py split_sentences(): NOT lossless (it strips each
piece and appends a "."), which is why P-ALIGN's training prefixes contain ".." at most cuts.
"""
import re

# The paper's rule: cut at backtracking cues.
CUE_PATTERN = (
    r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|\n\nBut just to|\n\nHowever"
    r"|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)"
)
# The SegmentSelectiveSFT fork's default: every "\n\n" opens a new segment.
PARAGRAPH_PATTERN = r"(\n\n)"

SEGMENT_PATTERNS = {"cue": CUE_PATTERN, "paragraph": PARAGRAPH_PATTERN}


def split_segments(text: str, mode: str = "paragraph") -> list[str]:
    if mode not in SEGMENT_PATTERNS:
        raise ValueError(f"mode must be one of {sorted(SEGMENT_PATTERNS)}, got {mode!r}")
    parts = re.split(SEGMENT_PATTERNS[mode], text)
    segments = [parts[0]]
    for index in range(1, len(parts), 2):
        segments.append(parts[index] + parts[index + 1])
    return _drop_empty(segments, text)


def _drop_empty(segments: list[str], text: str) -> list[str]:
    merged: list[str] = []
    for segment in segments:
        if segment == "":
            continue
        if merged and merged[-1] == "":
            merged[-1] = segment
        else:
            merged.append(segment)
    if not merged:
        return [text]
    assert "".join(merged) == text, "split_segments lost characters"
    return merged


def paragraph_segments(text: str) -> list[str]:
    return split_segments(text, "paragraph")


def cue_segments(text: str) -> list[str]:
    return split_segments(text, "cue")


def palign_sentences(text: str) -> list[str]:
    sentences = text.split(". ")
    return [
        sentence.strip() + "." if not sentence.endswith(".") else sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
