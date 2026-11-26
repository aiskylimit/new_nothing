import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from sgl.data.prepare import build_record
from sgl.diagnostics import answer_steps

TOKENIZER = str(Path(__file__).parent / "regression" / "assets" / "tokenizer")


def test_mentions_are_standalone_terms_and_boxed_matches_the_answer_only():
    text = "We get 120 and 3.12, then 12. So \\boxed{ 12 } not \\boxed{13}."
    assert [text[x:y] for x, y in answer_steps.mention_spans(text, "12")] == ["12", "12"]
    assert [text[x:y] for x, y in answer_steps.boxed_spans(text, "12")] == [" 12 "]


def test_shares_are_computed_over_step_tokens_and_loss_weights(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    response = "First we add the numbers. That gives 42 here. Check the work again. The answer is \\boxed{42}."
    record = build_record(tokenizer, "What is 40+2?", response, max_tokens=4096)
    record["id"] = 0
    assert len(record["steps"]) == 4
    spans = [(step["token_start"], step["token_end"]) for step in record["steps"]]
    lengths = [end - start for start, end in spans]
    weights = [0.0] * len(record["input_ids"])
    for (start, end), weight in zip(spans, [1.0, 1.0, 1.0, 2.0]):
        weights[start:end] = [weight] * (end - start)

    (tmp_path / "data.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "weights.jsonl").write_text(json.dumps({"id": 0, "loss_weights": weights}) + "\n")
    (tmp_path / "gains.json").write_text(json.dumps({"0": [1.0, 3.0, -5.0, 4.0]}))
    answer_steps.main([
        "--data-path", str(tmp_path / "data.jsonl"), "--tokenizer", TOKENIZER,
        "--weights", str(tmp_path / "weights.jsonl"), "--gains", str(tmp_path / "gains.json"),
        "--output-dir", str(tmp_path / "out"),
    ])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    total = sum(lengths)

    assert summary["boxed"]["answer_steps"] == 1 and summary["mention"]["answer_steps"] == 2
    assert summary["boxed"]["token_share"] == pytest.approx(lengths[3] / total)
    assert summary["mention"]["token_share"] == pytest.approx((lengths[1] + lengths[3]) / total)
    assert summary["boxed"]["weight_share"] == pytest.approx(2 * lengths[3] / (total + lengths[3]))
    assert summary["boxed"]["gain_share"] == pytest.approx(4 / 8)  # negative gain left out
    assert 0 < summary["mention"]["answer_tokens"] < summary["mention"]["answer_step_tokens"]
    mention = summary["mention"]
    assert mention["non_answer_tokens"] == mention["answer_step_tokens"] - mention["answer_tokens"]
    # "That gives ... here." has real words besides 42; "The answer is \\boxed{42}." has only the announcement
    assert 0 < mention["other_tokens"] < mention["non_answer_tokens"]
    assert summary["boxed"]["other_tokens"] == 0 and summary["boxed"]["bare_answer_step_share"] == 1.0
    assert 0 < summary["boxed"]["answer_token_weight_share"] < summary["boxed"]["weight_share"]
    assert "<mark>42</mark>" in (tmp_path / "out" / "answer-steps.html").read_text()
    assert (tmp_path / "out" / "position.png").exists()
    rows = (tmp_path / "out" / "answer-steps.jsonl").read_text().splitlines()
    assert [json.loads(row)["index"] for row in rows] == [1, 3]


def test_a_different_tokenization_is_rejected():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    record = build_record(tokenizer, "q", "The answer is \\boxed{1}.", max_tokens=4096)
    record["id"] = 0
    record["input_ids"] = [0] * len(record["input_ids"])
    with pytest.raises(ValueError, match="does not reproduce"):
        answer_steps.analyze_record(record, tokenizer, None, None)
