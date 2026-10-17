import json
import random
from pathlib import Path

import pytest
from reference import PALIGN, SSFT, load

from sgl.segment import SEGMENTERS, palign_sentences, split, split_segments

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
SAMPLE = json.loads((ASSETS / "palign_sample.jsonl").open().readline())["output"]
TEXTS = [
    SAMPLE,
    "\n\nleading break\n\n\n\nfour newlines\n\nWait, a cue.\n\nAlternatively x\n\nHowever y",
    "no breaks at all",
    "Wait\n\nWait\n\n\n\nBut wait, no.\n\n",
    "",
]


@pytest.fixture(scope="module")
def upstream():
    return load(SSFT / "segment_utils.py", "upstream_segment_utils")


@pytest.mark.parametrize("mode", ["paragraph", "cue"])
@pytest.mark.parametrize("text", TEXTS)
def test_ssft_segments_match_upstream(upstream, mode, text):
    assert split_segments(text, mode) == upstream.split_segments(text, mode)


def test_palign_sentences_match_upstream():
    upstream = load(PALIGN / "binary_search.py", "upstream_binary_search")
    rng = random.Random(0)
    texts = TEXTS + ["a. b.. c. ", ". . x.", "end without period", "x." * 5]
    texts += ["".join(rng.choice("ab. \n") for _ in range(60)) for _ in range(200)]
    for text in texts:
        assert palign_sentences(text) == upstream.split_sentences(text), repr(text)


@pytest.mark.parametrize("mode", sorted(SEGMENTERS))
@pytest.mark.parametrize("text", TEXTS)
def test_lossless_segmenters_reproduce_the_text(mode, text):
    pieces = split(text, mode)
    assert "".join(pieces) == text
    assert all(pieces)


def test_unknown_segmenter():
    with pytest.raises(ValueError):
        split("x", "words")
