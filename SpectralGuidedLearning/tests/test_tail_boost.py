import json

import pytest

from sgl.transforms.tail_boost import boost_tail, main


def test_boost_tail_scales_last_supervised_tokens_and_keeps_mass():
    mask = [0, 0, 1, 1, 1, 1, 1, 0]
    weights = boost_tail(mask, None, tail_tokens=2, boost=3.0)
    assert weights[0] == weights[1] == weights[7] == 0.0
    assert weights[5] / weights[2] == pytest.approx(3.0)
    assert weights[6] / weights[4] == pytest.approx(3.0)
    assert sum(weights) == pytest.approx(5.0)


def test_boost_tail_applies_on_top_of_existing_weights():
    mask, base = [1, 1, 1, 1], [2.0, 1.0, 0.5, 0.5]
    weights = boost_tail(mask, base, tail_tokens=1, boost=2.0)
    assert weights[3] / weights[2] == pytest.approx(2.0)
    assert weights[0] / weights[1] == pytest.approx(2.0)
    assert sum(weights) == pytest.approx(sum(base))


def test_boost_one_or_zero_tail_is_identity():
    mask = [0, 1, 1, 1]
    assert boost_tail(mask, None, tail_tokens=2, boost=1.0) == pytest.approx([0.0, 1.0, 1.0, 1.0])
    assert boost_tail(mask, None, tail_tokens=0, boost=5.0) == pytest.approx([0.0, 1.0, 1.0, 1.0])


def test_tail_longer_than_response_is_uniform():
    assert boost_tail([1, 1, 0], None, tail_tokens=10, boost=4.0) == pytest.approx([1.0, 1.0, 0.0])


def test_main_writes_boosted_dataset(tmp_path, capsys):
    rows = [{"id": 1, "input_ids": [5, 6, 7, 8], "loss_mask": [0, 1, 1, 1]}]
    source = tmp_path / "train-vanilla.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    main(["--data-path", str(source), "--output", str(tmp_path / "out.jsonl"), "--tail-tokens", "1", "--boost", "2"])
    out = json.loads((tmp_path / "out.jsonl").read_text())
    assert out["input_ids"] == rows[0]["input_ids"] and out["loss_mask"] == rows[0]["loss_mask"]
    assert out["loss_weights"] == pytest.approx([0.0, 0.75, 0.75, 1.5])
    assert "mean mass ratio 1.000000" in capsys.readouterr().out
