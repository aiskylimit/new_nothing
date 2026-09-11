"""Tests for iwc_diagnostics.py against synthetic (no-GPU) data.

Mirrors the record factory in test_build_masks.py; strengths/entropies are supplied directly
since no real gradient_capture.py output is needed to test the diagnostics maths/plots.
"""

import json

import pandas as pd
import pytest

from iwc_diagnostics import (
    compute_correlations,
    main,
    partial_pearson,
    pearson,
    per_sample_diagnostics,
    per_trajectory_correlation_distribution,
    plot_entropy_alpha_binned,
    plot_weight_mass_ratio_hist,
    spearman,
    within_trajectory_percentile,
)


def _record(record_id: int, step_lengths: list[int]) -> dict:
    spans, cursor = [], 0
    for length in step_lengths:
        spans.append({"token_start": cursor, "token_end": cursor + length})
        cursor += length
    return {
        "id": record_id,
        "input_ids": list(range(cursor + 1)),
        "response_token_span": [0, cursor],
        "steps": spans,
    }


def test_iwc_stable_weight_mass_ratio_is_one_at_full_interpolation():
    records = {0: _record(0, [10, 20, 5, 15])}
    signals = {0: ([4.0, 3.0, 2.0, 1.0], [0.1, 0.9, 0.5, 0.2])}

    samples = per_sample_diagnostics(
        records, signals, threshold=1.0, temperature=1.0, interpolation=1.0, clip=2.0, epsilon=1e-8
    )

    assert samples[0]["weight_mass_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_pearson_matches_known_perfect_correlation():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_none_for_constant_input():
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_partial_pearson_removes_confound():
    # y is exactly x + z; controlling for z should zero out the raw x-y correlation's
    # dependence on z, leaving a strong residual link between x and y.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    z = [5.0, 1.0, 4.0, 2.0, 3.0]
    y = [a + b for a, b in zip(x, z)]

    assert partial_pearson(x, y, z) == pytest.approx(1.0, abs=1e-6)


def test_compute_correlations_notes_missing_nll_and_skips_density_without_tokenizer():
    records = {0: _record(0, [10, 20, 5, 15])}
    signals = {0: ([4.0, 3.0, 2.0, 1.0], [0.1, 0.9, 0.5, 0.2])}
    samples = per_sample_diagnostics(
        records, signals, threshold=1.0, temperature=1.0, interpolation=1.0, clip=2.0, epsilon=1e-8
    )

    result = compute_correlations(samples, records, tokenizer_name=None)

    assert result["entropy_vs_nll"] is None
    assert "not store per-step NLL" in result["entropy_vs_nll_note"]
    assert result["entropy_vs_numeric_density"] is None


def test_spearman_is_scale_invariant_unlike_pearson():
    # A monotonic but non-linear map: pearson is imperfect, spearman must be exactly 1.
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]  # y = x^2, strictly increasing
    assert spearman(x, y) == pytest.approx(1.0)
    assert pearson(x, y) < 1.0


def test_spearman_averages_ranks_on_ties():
    # [1,1,2] -> ranks [1.5, 1.5, 3]; must not crash and must still see the perfect trend.
    assert spearman([1, 1, 2], [10, 10, 20]) == pytest.approx(1.0)


def test_within_trajectory_percentile_spans_zero_to_one():
    percentiles = within_trajectory_percentile([0.1, 0.9, 0.5, 0.2])
    assert min(percentiles) == pytest.approx(0.0)
    assert max(percentiles) == pytest.approx(1.0)
    assert len(percentiles) == 4


def test_per_trajectory_correlation_distribution_skips_short_trajectories():
    samples = [
        {"x": [1.0, 2.0], "y": [1.0, 2.0]},  # only 2 points, below default min_selected_steps=3
        {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]},  # perfect positive, qualifies
    ]

    result = per_trajectory_correlation_distribution(samples, "x", "y")

    assert result["n_trajectories"] == 1
    assert result["n_skipped"] == 1
    assert result["mean"] == pytest.approx(1.0)


def test_entropy_alpha_binned_plot_runs_on_synthetic_data(tmp_path):
    records = {0: _record(0, [10, 20, 5, 15]), 1: _record(1, [8, 8, 8])}
    signals = {
        0: ([4.0, 3.0, 2.0, 1.0], [0.1, 0.9, 0.5, 0.2]),
        1: ([1.0, 2.0, 3.0], [0.3, 0.6, 0.9]),
    }
    samples = per_sample_diagnostics(
        records, signals, threshold=0.9, temperature=1.0, interpolation=1.0, clip=2.0, epsilon=1e-8
    )

    plot_entropy_alpha_binned(samples, tmp_path)

    assert (tmp_path / "entropy_alpha_binned.png").exists()


def test_weight_mass_ratio_hist_survives_a_perfectly_uniform_corpus(tmp_path):
    # Every ratio is exactly 1.0 (the real qwen25-7b capture produced exactly this at scale):
    # np.histogram with a zero-width range used to raise "too many bins for data range".
    samples = [{"weight_mass_ratio": 1.0} for _ in range(1000)]

    plot_weight_mass_ratio_hist(samples, tmp_path)

    assert (tmp_path / "weight_mass_ratio_hist.png").exists()


def test_main_end_to_end_writes_expected_artifacts(tmp_path, monkeypatch):
    records = [_record(0, [10, 20, 5, 15]), _record(1, [8, 8, 8])]
    data_path = tmp_path / "train-segmented.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    frame = pd.DataFrame([
        {"id": 0, "step_strengths": [4.0, 3.0, 2.0, 1.0], "step_entropies": [0.1, 0.9, 0.5, 0.2]},
        {"id": 1, "step_strengths": [1.0, 2.0, 3.0], "step_entropies": [0.3, 0.6, 0.9]},
    ])
    strengths_path = tmp_path / "spectral-strengths.parquet"
    frame.to_parquet(strengths_path)

    output_dir = tmp_path / "diagnostics"
    monkeypatch.setattr(
        "sys.argv",
        [
            "iwc_diagnostics.py",
            "--data-path", str(data_path),
            "--strengths", str(strengths_path),
            "--output-dir", str(output_dir),
            "--energy-threshold-p", "0.9",
            "--n-traces", "2",
        ],
    )

    main()

    assert (output_dir / "weight_mass_ratio_hist.png").exists()
    assert (output_dir / "length_vs_weight_scatter.png").exists()
    assert (output_dir / "trace_heatmaps.png").exists()
    assert (output_dir / "entropy_alpha_binned.png").exists()
    correlations = json.loads((output_dir / "correlations.json").read_text())
    assert correlations["n_selected_steps"] > 0
    assert "entropy_vs_stable_weight_per_trajectory" in correlations
    assert "entropy_vs_length_per_trajectory" in correlations
