"""Tests for the p-sweep (drop-ratio sensitivity check restored after the review found
`energy_threshold_p` had been mislabeled as a paper-published value — it is not; see
docs/system-architecture.md and plans/reports/review-260805-plan-code-vs-paper.md)."""

import json

from build_masks import emit_masked_dataset, sweep_thresholds


def _record(record_id: int, step_lengths: list[int]) -> dict:
    """A record whose steps are each `length` tokens long, back to back, no prompt."""
    spans, cursor = [], 0
    for length in step_lengths:
        spans.append({"token_start": cursor, "token_end": cursor + length})
        cursor += length
    return {
        "id": record_id,
        "input_ids": list(range(cursor)),
        "steps": spans,
    }


def test_sweep_reports_monotonically_increasing_retention_as_p_grows():
    records = [_record(0, [10, 10, 10, 10])]
    strengths = {0: [4.0, 3.0, 2.0, 1.0]}  # strictly decreasing -> unambiguous ranking

    result = sweep_thresholds(records, strengths, [0.4, 0.7, 1.0])

    token_drops = [result[p]["token_drop_mean"] for p in [0.4, 0.7, 1.0]]
    assert token_drops[0] >= token_drops[1] >= token_drops[2] == 0.0


def test_sweep_matches_hand_computed_selection_at_one_threshold():
    # total strength = 10; p=0.75 needs cumulative >= 7.5 -> steps ranked [0](4),[1](3),[2](2),[3](1)
    # 4 -> 7 -> 9, so keep steps 0,1,2 (cum=9 >= 7.5); drops step 3 only (10 of 40 tokens = 25%)
    records = [_record(0, [10, 10, 10, 10])]
    strengths = {0: [4.0, 3.0, 2.0, 1.0]}

    result = sweep_thresholds(records, strengths, [0.75])

    assert result[0.75]["token_drop_mean"] == 0.25
    assert result[0.75]["step_drop_mean"] == 0.25


def test_sweep_never_touches_the_filesystem(monkeypatch):
    """Sweep is a read-only sensitivity check -- it must not write train-*.jsonl."""
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sweep opened a file")))
    records = [_record(0, [5, 5])]
    strengths = {0: [1.0, 1.0]}

    sweep_thresholds(records, strengths, [0.5, 1.0])  # would raise if it opened anything


def test_sweep_and_emit_agree_on_the_same_threshold(tmp_path):
    """The sweep's reported stats must match what emit_masked_dataset actually writes."""
    records = [_record(0, [10, 10, 10])]
    strengths = {0: [3.0, 2.0, 1.0]}
    output_path = tmp_path / "train-spectral.jsonl"

    swept = sweep_thresholds(records, strengths, [0.6])[0.6]
    emitted_stats = emit_masked_dataset(records, strengths, 0.6, output_path)[0]

    assert swept["token_drop_mean"] == emitted_stats["token_drop"]
    assert swept["step_drop_mean"] == emitted_stats["step_drop"]

    written = json.loads(output_path.read_text().strip())
    assert sum(written["loss_mask"]) == 30 - emitted_stats["n_tokens_dropped"]
