"""Segmentation tests use a whitespace-free character tokenizer so token offsets are
exactly predictable; the real-tokenizer round-trip is checked by data_prep.py itself."""

import re

from segmentation import segment_response_token_spans, solution_step_start, split_into_step_texts


class CharTokenizer:
    """Minimal stand-in: one token per character."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoding = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            encoding["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return encoding


class MergingTokenizer:
    """Stand-in for byte-level BPE: whitespace merges into the word that follows it.

    Reproduces the property that makes piecewise tokenization wrong — tokenizing " Foo"
    yields one token, but tokenizing " " and "Foo" separately yields two.
    """

    _PRETOKEN = re.compile(r"\s*\S+|\s+")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(self._PRETOKEN.finditer(text))
        encoding = {"input_ids": [hash(m.group()) % 50000 for m in matches]}
        if return_offsets_mapping:
            encoding["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return encoding


def test_split_preserves_original_text():
    response = "Step one. Step two.\n\nStep three."
    assert "".join(split_into_step_texts(response)) == response


def test_split_on_sentence_boundaries():
    assert split_into_step_texts("Step one. Step two. Step three.") == [
        "Step one. ",
        "Step two. ",
        "Step three.",
    ]


def test_split_on_newline_after_punctuation():
    assert len(split_into_step_texts("First part?\nSecond part!\nThird part.")) == 3


def test_split_after_closing_brace_or_bracket():
    assert len(split_into_step_texts(r"So \boxed{7} Therefore we stop.")) == 2
    assert len(split_into_step_texts(r"Hence \] Now continue.")) == 2


def test_no_split_without_capital_letter():
    assert len(split_into_step_texts("value is 3. then we continue")) == 1


def test_no_split_without_whitespace():
    assert len(split_into_step_texts("3.5 is the answer")) == 1


def test_empty_response_yields_no_steps():
    assert split_into_step_texts("") == []


def test_spans_are_contiguous_and_offset_by_prompt():
    tokenizer = CharTokenizer()
    prompt_ids = [1, 2, 3]
    response = "Aaaaaaaaaa. Bbbbbbbbbb. Cccccccccc."

    response_ids, spans = segment_response_token_spans(tokenizer, prompt_ids, response)

    assert len(spans) == 3
    assert spans[0][0] == len(prompt_ids)
    assert spans[-1][1] == len(prompt_ids) + len(response_ids)
    for previous, following in zip(spans, spans[1:]):
        assert previous[1] == following[0]  # no gaps, no overlaps


def test_short_steps_are_kept_not_merged():
    tokenizer = CharTokenizer()
    response = "Aaaaaaaaaaaa. B. Cccccccccccc."

    _, spans = segment_response_token_spans(tokenizer, [], response)

    assert len(spans) == 3  # short middle step survives — no merging
    assert all(end > start for start, end in spans)


def test_unsplittable_response_yields_single_span():
    tokenizer = CharTokenizer()
    _, spans = segment_response_token_spans(tokenizer, [], "one long line without boundaries")
    assert len(spans) == 1


def test_ids_are_the_canonical_tokenization_not_a_piecewise_concat():
    """Guards the whole point of offset-based mapping: the model must see the same ids a
    plain tokenizer call produces, not ids stitched together from separately encoded steps."""
    tokenizer = MergingTokenizer()
    response = "First step is done. Second step follows. Third step ends it."

    response_ids, _ = segment_response_token_spans(tokenizer, [], response)

    canonical = tokenizer(response)["input_ids"]
    piecewise = sum(
        (tokenizer(piece)["input_ids"] for piece in split_into_step_texts(response)), []
    )
    assert response_ids == canonical
    assert len(piecewise) > len(canonical)  # the bug this test exists to prevent


def test_solution_step_start_finds_the_tag_bearing_step():
    response = "First reasoning bit. More reasoning.\n</think>\n\nHere is the answer."
    steps = split_into_step_texts(response)

    boundary = solution_step_start(steps)

    assert boundary is not None
    assert "</think>" in steps[boundary]


def test_solution_step_start_returns_none_without_a_closing_tag():
    steps = split_into_step_texts("First reasoning bit. Second bit. Final answer is 7.")
    assert solution_step_start(steps) is None


def test_solution_step_start_matches_a_tag_split_mid_step():
    """The tag rarely lands on a sentence boundary -- it should still be found inside
    whichever step contains it, not silently missed."""
    steps = ["No boundary here so </think> stays embedded mid-sentence and continues on."]
    assert solution_step_start(steps) == 0


def test_spans_stay_contiguous_when_a_token_straddles_a_boundary():
    tokenizer = MergingTokenizer()
    response = "First step is done. Second step follows. Third step ends it."

    response_ids, spans = segment_response_token_spans(tokenizer, [3, 4], response)

    assert len(spans) == 3
    assert spans[0][0] == 2 and spans[-1][1] == 2 + len(response_ids)
    for previous, following in zip(spans, spans[1:]):
        assert previous[1] == following[0]
