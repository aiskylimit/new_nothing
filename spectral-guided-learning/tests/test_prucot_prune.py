from prucot_prune import (
    build_virtual_id_map,
    candidate_step_indices,
    emit_prucot_record,
    format_step_table,
    median_gate,
    parse_prune_response,
    reconstruct_pruned_text,
)


def test_median_gate_true_when_median_above_threshold():
    assert median_gate([0.9, 1.0, 1.0], threshold=0.8) is True


def test_median_gate_false_at_or_below_threshold():
    assert median_gate([0.1, 0.2, 0.3], threshold=0.8) is False


def test_median_gate_false_on_empty_weights():
    assert median_gate([], threshold=0.8) is False


def test_candidate_step_indices_only_below_threshold():
    assert candidate_step_indices([0.9, 0.1, 0.4, 0.9], threshold=0.5) == [1, 2]


def test_candidate_step_indices_never_includes_the_first_or_last_scored_step():
    """The first scored step usually carries the response's opening "<think>" tag, and the
    last is a uniform safety margin -- neither is ever a pruning candidate, even when its own
    weight is below threshold."""
    assert candidate_step_indices([0.1, 0.1, 0.1, 0.1], threshold=0.5) == [1, 2]


def test_candidate_step_indices_empty_when_no_weights():
    assert candidate_step_indices([], threshold=0.5) == []


def test_candidate_step_indices_empty_when_fewer_than_three_scored_steps():
    """Excluding both ends of a 1- or 2-step chain would leave nothing meaningful to protect
    or prune -- treat the whole tiny chain as protected instead."""
    assert candidate_step_indices([0.1], threshold=0.5) == []
    assert candidate_step_indices([0.1, 0.1], threshold=0.5) == []


def test_build_virtual_id_map_is_one_indexed_and_sorted():
    assert build_virtual_id_map([5, 1, 3]) == {1: 1, 3: 2, 5: 3}


def test_format_step_table_blanks_non_candidate_rows():
    virtual_ids = {1: 1}
    table = format_step_table(["first", "second", "third"], virtual_ids)

    lines = table.split("\n")
    assert lines[0] == "|  | first |"
    assert lines[1] == "| 1 | second |"
    assert lines[2] == "|  | third |"


def test_format_step_table_escapes_pipe_and_flattens_newlines():
    table = format_step_table(["a | b\nc"], {0: 1})
    assert table == "| 1 | a \\| b c |"


def test_parse_prune_response_extracts_pruned_virtual_ids():
    text = '{"1": {"reasoning": "x", "prune": true}, "2": {"reasoning": "y", "prune": false}}'
    pruned, status = parse_prune_response(text)
    assert pruned == [1]
    assert status == "ok"


def test_parse_prune_response_handles_python_style_booleans():
    """The ported system prompt's own example uses Python True/False -- json_repair must cope."""
    text = '{"1": {"reasoning": "x", "prune": True}}'
    pruned, _ = parse_prune_response(text)
    assert pruned == [1]


def test_parse_prune_response_rejects_non_dict_payload():
    pruned, status = parse_prune_response("[1, 2, 3]")
    assert pruned is None
    assert status.startswith("parse_failed")


def test_parse_prune_response_ignores_non_integer_keys():
    text = '{"note": "ignore me", "1": {"prune": true}}'
    pruned, _ = parse_prune_response(text)
    assert pruned == [1]


def test_reconstruct_pruned_text_keeps_order_and_drops_pruned_indices():
    steps = ["Step one. ", "Step two. ", "Step three."]
    assert reconstruct_pruned_text(steps, {1}) == "Step one. Step three."


def test_reconstruct_pruned_text_empty_prune_set_reproduces_original():
    steps = ["a", "b", "c"]
    assert reconstruct_pruned_text(steps, set()) == "abc"


def test_emit_prucot_record_reuses_input_ids_when_nothing_pruned():
    record = {"id": 7, "input_ids": [1, 2, 3, 4, 5], "response_token_span": [2, 5]}

    row = emit_prucot_record(record, prune_indices=set())

    assert row == {"id": 7, "input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 1, 1, 1]}


class _StubTokenizer:
    """Maps each character to a synthetic token id so retokenization is easy to assert on."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def test_emit_prucot_record_retokenizes_the_shortened_response_when_pruned():
    record = {"id": 3, "input_ids": [100, 100], "response_token_span": [1, 2]}
    step_texts = ["A", "B"]

    row = emit_prucot_record(
        record, prune_indices={1}, n_scored=2, student_tokenizer=_StubTokenizer(), step_texts=step_texts,
    )

    assert row["input_ids"] == [100, ord("A")]
    assert row["loss_mask"] == [0, 1]


def test_emit_prucot_record_never_prunes_the_solution_tail_past_n_scored():
    """prune_indices only ever index into step_texts[:n_scored] (the scored chain-of-thought);
    anything from n_scored onward -- the real solution, when prucot_weight.py found a split --
    must always survive untouched in the reconstruction."""
    record = {"id": 4, "input_ids": [100, 100, 100], "response_token_span": [1, 3]}
    step_texts = ["A", "B", "SOLUTION"]

    row = emit_prucot_record(
        record, prune_indices={0}, n_scored=2, student_tokenizer=_StubTokenizer(), step_texts=step_texts,
    )

    expected_text = "B" + "SOLUTION"
    assert row["input_ids"] == [100] + [ord(char) for char in expected_text]
    assert row["loss_mask"] == [0] + [1] * len(expected_text)
