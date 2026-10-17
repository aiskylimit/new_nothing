"""\\n\\n-step structure for L_trans. A one-char-per-token tokenizer makes offsets exact."""

from sgl.transforms.step_transitions import (
    build_transition_fields,
    cot_char_span,
    merge_short_steps,
    split_cot_char_cuts,
)


def _char_token_starts(text: str) -> list[int]:
    return list(range(len(text)))


def _fields(response: str, prompt_len: int = 3, min_step_tokens: int = 2, stop: int = 1) -> dict:
    return build_transition_fields(
        response,
        _char_token_starts(response),
        prompt_len,
        prompt_len + len(response) + stop,
        min_step_tokens,
    )


def test_cot_span_handles_every_response_layout():
    assert cot_char_span("<think>\nabc\n</think>\n\nans") == (len("<think>"), len("<think>\nabc\n"))
    assert cot_char_span("abc\n</think>\n\nans") == (0, len("abc\n"))  # prompt opened <think>
    assert cot_char_span("abc\n\n\nans") == (0, len("abc\n\n\nans"))  # no tags: whole response


def test_steps_keep_their_trailing_break_and_multi_newlines_are_one_boundary():
    response = "<think>\naaaa\n\nbbbb\n\n\ncccc\n</think>\n\nzz"
    cuts = split_cot_char_cuts(response)
    start, end = cot_char_span(response)
    pieces = [response[a:b] for a, b in zip(cuts, cuts[1:])]
    assert pieces == ["\naaaa\n\n", "bbbb\n\n\n", "cccc\n"]
    assert cuts[0] == start and cuts[-1] == end


def test_fields_mark_prompt_answer_and_stop_as_outside_cot():
    response = "<think>\naaaa\n\nbbbb\n\ncccc\n</think>\n\nzz"
    prompt_len = 3
    fields = _fields(response, prompt_len=prompt_len)
    step_id = fields["step_id"]
    assert step_id[:prompt_len] == [-1] * prompt_len
    cot_start, cot_end = cot_char_span(response)
    assert all(step_id[prompt_len + i] >= 0 for i in range(cot_start, cot_end))
    assert all(step_id[prompt_len + i] == -1 for i in range(cot_end, len(response)))
    assert step_id[-1] == -1  # stop token
    assert fields["num_steps"] == 3
    # step_end[i] is the token holding the second "\n" of the break that closes step i
    for i, end in enumerate(fields["step_end"][:-1]):
        assert response[end - prompt_len] == "\n" and step_id[end] == i and step_id[end + 1] == i + 1
    assert fields["pair_src"] == [0, 1]


def test_source_position_precedes_every_target_token():
    """Causality of the objective: s_i must sit strictly before every token of step i+1."""
    response = "<think>\naaaa\n\nbbbb\n\ncccc\n\ndddd\n</think>\n\nzz"
    fields = _fields(response)
    for i in fields["pair_src"]:
        target_positions = [t for t, s in enumerate(fields["step_id"]) if s == i + 1]
        assert fields["step_end"][i] < min(target_positions)


def test_short_steps_merge_into_previous_and_first_into_next():
    assert merge_short_steps([(0, 10), (10, 12), (12, 20)], 5) == [(0, 12), (12, 20)]
    assert merge_short_steps([(0, 2), (2, 10), (10, 20)], 5) == [(0, 10), (10, 20)]
    assert merge_short_steps([(0, 10), (10, 12), (12, 13), (13, 20)], 5) == [(0, 13), (13, 20)]
    assert merge_short_steps([], 5) == []


def test_merged_fields_have_no_step_below_minimum_except_when_whole_cot_is_short():
    response = "<think>\naaaaaaaa\n\nb\n\ncccccccc\n\nd\n</think>\n\nzz"
    fields = _fields(response, min_step_tokens=4)
    lengths = [fields["step_id"].count(k) for k in range(fields["num_steps"])]
    assert fields["num_steps"] == 2 and all(length >= 4 for length in lengths)
    assert fields["pair_src"] == [0]

    tiny = _fields("<think>\nab\n</think>\n\nzz", min_step_tokens=4)
    assert tiny["num_steps"] == 1 and tiny["pair_src"] == []


def test_empty_cot_yields_no_steps():
    fields = _fields("<think>\n\n</think>\n\nzz")
    assert fields["num_steps"] == 0 and fields["step_end"] == [] and fields["pair_src"] == []
    assert set(fields["step_id"]) == {-1}
