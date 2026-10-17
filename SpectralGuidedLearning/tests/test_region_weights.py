import json

import pandas as pd
import pytest

from sgl.allocation import build
from sgl.allocation.build import build_example
from sgl.allocation.iwc import stable_iwc_step_weights
from sgl.allocation.region import (
    CONTINUATION,
    JUNCTION,
    PREFIX,
    region_budget,
    region_gain_ratio,
    region_stable_step_weights,
    step_regions,
)

P, C, J = PREFIX, CONTINUATION, JUNCTION


def mass(lengths, weights, keep=None):
    return sum(length * weight for index, (length, weight) in enumerate(zip(lengths, weights))
               if keep is None or keep[index])


def test_step_regions_marks_the_marker_step_as_junction():
    assert step_regions([(0, 3), (3, 5), (5, 9), (9, 12)], boundary=7) == [P, P, J, C]
    assert step_regions([(0, 3), (3, 5)], boundary=3) == [P, C]


def test_region_iwc_preserves_each_region_mass_separately():
    entropies = [3.0, 2.5, 0.1, 0.2, 0.3, 1.0]
    lengths = [4, 2, 7, 5, 3, 6]
    regions = [P, P, J, C, C, C]
    weights = region_stable_step_weights(entropies, lengths, regions, temperature=1.0)
    for region in (P, C):
        keep = [value == region for value in regions]
        assert mass(lengths, weights, keep) == pytest.approx(mass(lengths, [1.0] * 6, keep))
    assert weights[2] == 1.0


def test_global_iwc_moves_mass_to_a_high_entropy_prefix_but_region_iwc_does_not():
    entropies, lengths, regions = [3.0, 3.0, 0.5, 0.5], [5, 5, 5, 5], [P, P, C, C]
    keep = [r == P for r in regions]
    assert mass(lengths, stable_iwc_step_weights(entropies, lengths), keep) > 10.0
    assert mass(lengths, region_stable_step_weights(entropies, lengths, regions), keep) == pytest.approx(10.0)


def test_region_iwc_inside_one_region_matches_stable_iwc():
    entropies, lengths = [0.1, 0.9, 0.4], [2, 3, 4]
    assert region_stable_step_weights(entropies, lengths, [C, C, C], temperature=0.7) == pytest.approx(
        stable_iwc_step_weights(entropies, lengths, temperature=0.7))


def test_region_budget_sets_the_ratio_and_keeps_sample_mass():
    lengths, regions = [10, 2, 30], [P, J, C]
    budget = region_budget(lengths, regions, ratio=0.5)
    assert budget[0] / budget[2] == pytest.approx(0.5)
    assert budget[1] == 1.0
    assert mass(lengths, budget) == pytest.approx(42.0)


def test_region_budget_is_uniform_without_both_regions_or_at_ratio_one():
    assert region_budget([3, 4], [C, C], ratio=0.2) == [1.0, 1.0]
    assert region_budget([3, 4], [P, C], ratio=1.0) == pytest.approx([1.0, 1.0])


def test_region_gain_ratio_is_per_token_not_per_step():
    # Same per-step gains, but the prefix step is 4x longer -> 4x less gain per token.
    gain_prefix, gain_cont, ratio = region_gain_ratio([[1.0, 1.0]], [[8, 2]], [[P, C]])
    assert (gain_prefix, gain_cont, ratio) == pytest.approx((0.125, 0.5, 0.25))
    assert region_gain_ratio([[1.0, 1.0]], [[8, 2]], [[P, C]], gamma=0.5)[2] == pytest.approx(0.5)


def test_region_gain_ratio_rejects_non_positive_gain():
    with pytest.raises(ValueError):
        region_gain_ratio([[-1.0, 1.0]], [[2, 2]], [[P, C]])


RECORD = {
    "id": 3,
    "input_ids": list(range(13)),
    "response_token_span": [1, 12],
    "steps": [
        {"token_start": 1, "token_end": 4},
        {"token_start": 4, "token_end": 6},
        {"token_start": 6, "token_end": 8},
        {"token_start": 8, "token_end": 12},
    ],
}


@pytest.mark.parametrize("variant", ["region-iwc", "region-gain", "sarw"])
def test_region_variants_keep_the_sample_mass(variant):
    example, stats = build_example(
        RECORD, strengths=[1.0] * 4, entropies=[2.0, 1.0, 0.5, 0.1], threshold=1.0, variant=variant,
        temperature=1.0, interpolation=1.0, clip=2.0, epsilon=1e-8,
        regions=[P, P, J, C], region_ratio=0.5,
    )
    assert stats["weight_mass_ratio"] == pytest.approx(1.0)
    assert example["loss_weights"][12] == 1.0  # stop token stays uniform
    assert example["loss_weights"][0] == 0.0  # prompt stays unsupervised


def test_sarw_is_the_product_of_region_iwc_and_region_gain():
    def weights(variant):
        example, _ = build_example(
            RECORD, [1.0] * 4, [2.0, 1.0, 0.5, 0.1], 1.0, variant, 1.0, 1.0, 2.0, 1e-8,
            regions=[P, P, J, C], region_ratio=0.5,
        )
        return example["loss_weights"]

    product = [a * b for a, b in zip(weights("region-iwc"), weights("region-gain"))]
    assert weights("sarw") == pytest.approx(product)


def test_region_variant_without_regions_fails():
    with pytest.raises(ValueError):
        build_example(RECORD, [1.0] * 4, [0.0] * 4, 1.0, "sarw", 1.0, 1.0, 2.0, 1e-8)


def test_main_writes_a_mass_preserving_sarw_dataset(tmp_path, monkeypatch):
    records = [dict(RECORD, response="ab"), dict(RECORD, id=4, response="ab")]
    data = tmp_path / "train-segmented.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records))
    rows = lambda column: pd.DataFrame(  # noqa: E731
        [{"id": r["id"], "step_strengths": [1.0] * 4, "step_entropies": column} for r in records])
    rows([2.0, 1.0, 0.5, 0.1]).to_parquet(tmp_path / "entropy.parquet")
    rows([0.1, 0.1, 0.3, 2.0]).to_parquet(tmp_path / "gain.parquet")
    monkeypatch.setattr(build, "record_regions", lambda records, name: {r["id"]: [P, P, J, C] for r in records})

    build.main([
        "--data-path", str(data), "--strengths", str(tmp_path / "entropy.parquet"),
        "--variants", "sarw", "--energy-threshold-p", "1.0", "--tokenizer", "unused",
        "--region-signal", str(tmp_path / "gain.parquet"), "--output-name", "sarw", "--check-mass",
    ])
    summary = json.loads((tmp_path / "sarw-selection-stats.json").read_text())
    # prefix: 0.2 gain over 5 tokens; continuation: 2.0 over 4 tokens -> ratio 0.08
    assert summary["config"]["region_ratio"] == pytest.approx((0.2 / 5) / (2.0 / 4))
    assert (tmp_path / "train-sarw.jsonl").exists()


def test_region_gain_ratio_tail_fraction_drops_the_answer_steps():
    gains, lengths, regions = [[1.0, 1.0, 9.0]], [[2, 2, 1]], [[P, C, C]]
    assert region_gain_ratio(gains, lengths, regions)[2] == pytest.approx(0.5 / (10.0 / 3))
    assert region_gain_ratio(gains, lengths, regions, tail_fraction=0.1)[2] == pytest.approx(1.0)


def test_region_cont_drops_prefix_steps_and_keeps_unit_weights():
    example, stats = build_example(
        RECORD, [1.0] * 4, [2.0, 1.0, 0.5, 0.1], 1.0, "region-cont", 1.0, 1.0, 2.0, 1e-8,
        regions=[P, P, J, C],
    )
    # prefix steps (tokens 1..5) unsupervised; junction, continuation and stop token at weight 1
    assert example["loss_mask"] == [0] * 6 + [1] * 7
    assert example["loss_weights"] == [0.0] * 6 + [1.0] * 7
    assert stats["weight_mass_ratio"] == pytest.approx(1.0)
