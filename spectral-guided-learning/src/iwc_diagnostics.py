"""Pre-train IWC diagnostics (paper-adjacent controls, see docs/iwc-baselines.md).

CPU-only checks on the two artifacts gradient_capture.py/data_prep.py already produce
(train-segmented.jsonl, spectral-strengths.parquet) -- no train_sft.py run required. Reuses
the exact selection/weighting functions from step_selection.py and iwc_weights.py instead of
recomputing the maths, so a diagnostic can never silently drift from what build_iwc_datasets.py
actually ships.

Produces:
  - weight_mass_ratio_hist.png : per-sample weighted/selected token-mass ratio (IWC-Stable
    must cluster at 1.0 -- the token-mass invariance the docs call out).
  - length_vs_weight_scatter.png : selected-step length vs coefficient, IWC vs IWC-Stable
    side by side. IWC-Stable only guarantees Sigma(length * weight) = Sigma(length) (total
    token-mass invariance) -- it does NOT guarantee corr(length, weight) = 0, and in
    practice does not remove the length confound (both arms show similar r on this
    corpus). Read this plot as "does the per-step correlation shape differ", not as
    evidence Stable is confound-free.
  - trace_heatmaps.png : spectral strength / entropy / selected / IWC-Stable weight per step,
    for a handful of sampled traces. Unselected steps are NaN-masked (gray), not folded
    into the same 0-1 scale as real weights.
  - entropy_alpha_binned.png : the direct entropy-drives-weight check -- mean alpha per
    within-trajectory entropy percentile bin, IWC vs IWC-Stable.
  - correlations.json : entropy vs step length / relative position / numeric-token density
    (Pearson r, plus partial r controlling for length where relevant), entropy vs alpha
    (Spearman, per-trajectory), and per-trajectory correlation distributions -- steps
    within one trajectory are not independent, so a single pooled r understates its own
    uncertainty; see the pooled_correlation_caveat field.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iwc_weights import stable_iwc_step_weights, vanilla_iwc_step_weights
from segmentation import record_step_spans
from step_selection import select_steps_by_energy


def load_records(data_path: str) -> dict[int, dict]:
    with open(data_path) as handle:
        return {(record := json.loads(line))["id"]: record for line in handle}


def load_signals(strengths_path: str) -> dict[int, tuple[list[float], list[float]]]:
    frame = pd.read_parquet(strengths_path)
    if "step_entropies" not in frame.columns:
        raise ValueError(f"{strengths_path} has no step_entropies; rerun gradient_capture.py")
    return {int(row.id): (list(row.step_strengths), list(row.step_entropies)) for row in frame.itertuples()}


def per_sample_diagnostics(
    records: dict[int, dict],
    signals: dict[int, tuple[list[float], list[float]]],
    threshold: float,
    temperature: float,
    interpolation: float,
    clip: float,
    epsilon: float,
) -> list[dict]:
    """One entry per sample: selected steps' lengths/entropies/positions plus both IWC weightings."""
    samples = []
    for record_id, record in records.items():
        strengths, entropies = signals[record_id]
        step_spans = record_step_spans(record)
        if len(strengths) != len(step_spans) or len(entropies) != len(step_spans):
            raise ValueError(f"record {record_id}: step/signal count mismatch")
        lengths = [end - start for start, end in step_spans]
        selected = select_steps_by_energy(strengths, threshold)
        selected_lengths = [lengths[i] for i in selected]
        selected_entropies = [entropies[i] for i in selected]
        selected_positions = [i / max(len(step_spans) - 1, 1) for i in selected]

        iwc_weights = vanilla_iwc_step_weights(selected_entropies, selected_lengths, temperature)
        stable_weights = stable_iwc_step_weights(
            selected_entropies, selected_lengths, temperature, interpolation, clip, epsilon
        )
        selected_mass = sum(selected_lengths)
        weighted_mass = sum(l * w for l, w in zip(selected_lengths, stable_weights))

        samples.append({
            "id": record_id,
            "n_steps": len(step_spans),
            "strengths": strengths,
            "entropies": entropies,
            "selected": selected,
            "selected_lengths": selected_lengths,
            "selected_entropies": selected_entropies,
            "selected_positions": selected_positions,
            "iwc_weights": iwc_weights,
            "stable_weights": stable_weights,
            "weight_mass_ratio": weighted_mass / max(selected_mass, 1),
        })
    return samples


def numeric_density(tokenizer, record: dict, span: tuple[int, int]) -> float:
    start, end = span
    text = tokenizer.decode(record["input_ids"][start:end])
    return sum(char.isdigit() for char in text) / len(text) if text else 0.0


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def partial_pearson(x: list[float], y: list[float], z: list[float]) -> float | None:
    """Correlation of x,y with the linear effect of z removed from both."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    if rxy is None or rxz is None or ryz is None:
        return None
    denom = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
    return float((rxy - rxz * ryz) / denom) if denom > 1e-12 else None


def _rankdata(values: list[float]) -> np.ndarray:
    """Average ranks (1-indexed, ties averaged) -- scipy.stats.rankdata without scipy."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    raw_ranks = np.empty(len(array), dtype=np.float64)
    raw_ranks[order] = np.arange(1, len(array) + 1)
    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, raw_ranks)
    return (sums / counts)[inverse]


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(_rankdata(x).tolist(), _rankdata(y).tolist())


def within_trajectory_percentile(values: list[float]) -> list[float]:
    """Rank each value within its own trajectory to [0, 1] (0 = lowest entropy in-trace)."""
    if len(values) < 2:
        return [0.5] * len(values)
    ranks = _rankdata(values)
    return ((ranks - 1) / (len(values) - 1)).tolist()


def plot_weight_mass_ratio_hist(samples: list[dict], output_dir: Path) -> None:
    ratios = np.asarray([sample["weight_mass_ratio"] for sample in samples], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    spread = float(ratios.max() - ratios.min()) if len(ratios) else 0.0
    if spread < 1e-9:
        # The normalization in stable_iwc_step_weights is an exact renormalization
        # (weight * token_mass / weighted_mass), so with enough samples the ratio is
        # bit-identical to 1.0 across the corpus -- a real, good outcome, not a bug --
        # but a near-zero range breaks np.histogram's finite-bin-width computation.
        ax.axvline(1.0, color="red", linestyle="--", label="target = 1.0")
        ax.set_xlim(1.0 - 1e-6, 1.0 + 1e-6)
        ax.text(
            0.5, 0.5, f"all {len(ratios)} samples: ratio = 1.0 (spread={spread:.1e})",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.legend()
    else:
        ax.hist(ratios, bins=40)
        ax.axvline(1.0, color="red", linestyle="--", label="target = 1.0")
        ax.legend()
    ax.set_xlabel("IWC-Stable weighted_mass / selected_mass")
    ax.set_ylabel("samples")
    ax.set_title(f"Token-mass invariance (mean={ratios.mean():.6f}, std={ratios.std():.2e})")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_mass_ratio_hist.png", dpi=150)
    plt.close(fig)


def plot_entropy_alpha_binned(samples: list[dict], output_dir: Path, n_bins: int = 10) -> None:
    """Binned entropy-percentile vs mean alpha -- the direct "does entropy drive weight" check.

    Entropy is converted to its WITHIN-TRAJECTORY percentile (0=lowest entropy step in that
    trace, 1=highest) before pooling and binning: IWC-Stable z-scores entropy per trajectory,
    so raw entropy values are not on a comparable scale across different trajectories, and
    binning raw pooled entropy would mix that per-trajectory scale difference into the bins.
    Expectation for IWC-Stable: mean alpha should rise monotonically with the entropy
    percentile bin (except where clipping at +-`clip` flattens the top/bottom bins).
    """
    percentiles, iwc_w, stable_w = [], [], []
    for sample in samples:
        if len(sample["selected_entropies"]) < 2:
            continue
        percentiles.extend(within_trajectory_percentile(sample["selected_entropies"]))
        iwc_w.extend(sample["iwc_weights"])
        stable_w.extend(sample["stable_weights"])
    if not percentiles:
        return

    percentiles = np.asarray(percentiles)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(percentiles, bin_edges[1:-1]), 0, n_bins - 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for weights, label, marker in ((iwc_w, "IWC (vanilla)", "o"), (stable_w, "IWC-Stable", "s")):
        weights = np.asarray(weights)
        means = [weights[bin_index == b].mean() if np.any(bin_index == b) else np.nan for b in range(n_bins)]
        ax.plot(centers, means, marker=marker, label=label)
    # Same spectral mask, but no entropy reweighting: alpha is one at every
    # selected step. This is the exact spectral-only (lambda=0) baseline.
    ax.axhline(
        1.0, color="0.35", linestyle="--", linewidth=1.5,
        label="Spectral-only (uniform, λ=0)",
    )
    ax.set_xlabel("within-trajectory entropy percentile (selected steps only)")
    ax.set_ylabel("mean alpha")
    ax.set_title("Entropy -> alpha: binned by within-trajectory entropy rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_alpha_binned.png", dpi=150)
    plt.close(fig)


def plot_length_vs_weight(samples: list[dict], output_dir: Path) -> None:
    lengths = [l for sample in samples for l in sample["selected_lengths"]]
    iwc_w = [w for sample in samples for w in sample["iwc_weights"]]
    stable_w = [w for sample in samples for w in sample["stable_weights"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, weights, title in ((axes[0], iwc_w, "IWC (vanilla)"), (axes[1], stable_w, "IWC-Stable")):
        ax.hexbin(lengths, weights, gridsize=40, mincnt=1)
        r = pearson(lengths, weights)
        ax.set_title(f"{title}  (r={r:.3f})" if r is not None else title)
        ax.set_xlabel("selected step length (tokens)")
    axes[0].set_ylabel("coefficient alpha")
    fig.tight_layout()
    fig.savefig(output_dir / "length_vs_weight_scatter.png", dpi=150)
    plt.close(fig)


def plot_trace_heatmaps(samples: list[dict], output_dir: Path, n_traces: int, seed: int) -> None:
    candidates = [sample for sample in samples if sample["n_steps"] > 1]
    if not candidates:
        return
    rng = random.Random(seed)
    picked = rng.sample(candidates, k=min(n_traces, len(candidates)))

    # NaN-masked cells render as a distinct gray via cmap.set_bad, instead of being folded
    # into the same 0-1 scale as real weights (which made "not selected" visually
    # indistinguishable from "selected but low weight").
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(len(picked), 1, figsize=(10, 2.2 * len(picked)), squeeze=False)
    for row, sample in enumerate(picked):
        n = sample["n_steps"]
        selected_mask = np.zeros(n)
        selected_mask[sample["selected"]] = 1.0

        stable_full = np.full(n, np.nan)
        selected_index = np.asarray(sample["selected"])
        selected_weights = np.asarray(sample["stable_weights"], dtype=np.float64)
        # Normalize only over the selected steps' own weights -- not the whole (n,) array
        # of unselected zeros -- so the color scale reflects real weight variation only.
        span = selected_weights.max() - selected_weights.min() if len(selected_weights) else 0.0
        stable_full[selected_index] = (
            (selected_weights - selected_weights.min()) / span if span > 1e-12
            else np.zeros_like(selected_weights)
        )

        def normalize(values: np.ndarray) -> np.ndarray:
            span = values.max() - values.min()
            return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)

        grid = np.stack([
            normalize(np.asarray(sample["strengths"])),
            normalize(np.asarray(sample["entropies"])),
            selected_mask,
            stable_full,
        ])
        ax = axes[row][0]
        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["strength", "entropy", "selected", "stable weight"], fontsize=8)
        ax.set_title(f"sample id={sample['id']} ({n} steps)", fontsize=9)
    axes[-1][0].set_xlabel("step index")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_heatmaps.png", dpi=150)
    plt.close(fig)


def per_trajectory_correlation_distribution(
    samples: list[dict], x_key: str, y_key: str, min_selected_steps: int = 3, method=pearson,
) -> dict:
    """Distribution of one correlation computed per-sample, not pooled across all steps.

    Steps within the same trajectory are not independent observations, so a single
    correlation pooled over every selected step (282k+ of them, ~1000 trajectories)
    understates how uncertain that correlation really is. Reporting the per-trajectory
    distribution is the cluster-aware alternative: each trajectory contributes exactly one
    data point. Trajectories with fewer than min_selected_steps selected steps are skipped
    (too few points for a correlation to be meaningful) and counted separately.

    `method=spearman` is required for entropy-vs-alpha checks specifically: IWC-Stable
    z-scores entropy *within each trajectory* before weighting, so a step's raw entropy is
    not comparable across trajectories -- only a per-trajectory correlation (or a rank
    transform, see within_trajectory_percentile) is scale-consistent.
    """
    values = [
        r for sample in samples
        if len(sample[x_key]) >= min_selected_steps
        and (r := method(sample[x_key], sample[y_key])) is not None
    ]
    skipped = len(samples) - len(values)
    if not values:
        return {"n_trajectories": 0, "n_skipped": skipped, "mean": None, "median": None, "std": None}
    array = np.asarray(values)
    return {
        "n_trajectories": len(values),
        "n_skipped": skipped,
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def compute_correlations(
    samples: list[dict], records: dict[int, dict], tokenizer_name: str | None
) -> dict:
    entropies = [e for sample in samples for e in sample["selected_entropies"]]
    lengths = [l for sample in samples for l in sample["selected_lengths"]]
    positions = [p for sample in samples for p in sample["selected_positions"]]

    result = {
        "n_selected_steps": len(entropies),
        "pooled_correlation_caveat": (
            "entropy_vs_length/entropy_vs_position below pool every selected step across "
            "all trajectories; steps within one trajectory are not independent, so treat "
            "these as a quick check only -- see the *_per_trajectory distributions for the "
            "cluster-aware version (one r per trajectory)."
        ),
        "entropy_vs_length": pearson(entropies, lengths),
        "entropy_vs_length_per_trajectory": per_trajectory_correlation_distribution(
            samples, "selected_entropies", "selected_lengths"
        ),
        "entropy_vs_position": pearson(entropies, positions),
        "entropy_vs_position_per_trajectory": per_trajectory_correlation_distribution(
            samples, "selected_entropies", "selected_positions"
        ),
        # Spearman, not Pearson: IWC-Stable's construction (rank-preserving up to the +-clip
        # bound) predicts a monotonic, not necessarily linear, entropy-to-alpha relationship.
        # Per-trajectory only (no pooled version): entropy is z-scored within each trajectory
        # before weighting, so raw entropy is not on a comparable scale across trajectories --
        # see plot_entropy_alpha_binned's percentile transform for the pooled-friendly view.
        "entropy_vs_iwc_weight_per_trajectory": per_trajectory_correlation_distribution(
            samples, "selected_entropies", "iwc_weights", method=spearman
        ),
        "entropy_vs_stable_weight_per_trajectory": per_trajectory_correlation_distribution(
            samples, "selected_entropies", "stable_weights", method=spearman
        ),
        "entropy_vs_nll": None,
        "entropy_vs_nll_note": (
            "gradient_capture.py does not store per-step NLL, only predictive entropy and "
            "spectral strength -- computing it needs a separate frozen-model forward pass."
        ),
    }

    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        densities = []
        for sample in samples:
            record = records[sample["id"]]
            step_spans = record_step_spans(record)
            for index in sample["selected"]:
                densities.append(numeric_density(tokenizer, record, step_spans[index]))
        result["entropy_vs_numeric_density"] = pearson(entropies, densities)
        result["entropy_vs_numeric_density_partial_on_length"] = partial_pearson(
            entropies, densities, lengths
        )
    else:
        result["entropy_vs_numeric_density"] = None
        result["entropy_vs_numeric_density_note"] = "pass --tokenizer to compute (needs HF tokenizer files)"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl from data_prep.py")
    parser.add_argument("--strengths", required=True, help="spectral-strengths.parquet from gradient_capture.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--energy-threshold-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--interpolation", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--tokenizer", help="HF tokenizer id/path, for numeric-token-density correlation")
    parser.add_argument("--n-traces", type=int, default=6, help="sample traces to draw in the heatmap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data_path)
    signals = load_signals(args.strengths)
    missing = [record_id for record_id in records if record_id not in signals]
    if missing:
        raise ValueError(f"spectral signals missing for {len(missing)} records (first: {missing[0]})")

    samples = per_sample_diagnostics(
        records, signals, args.energy_threshold_p, args.temperature,
        args.interpolation, args.clip, args.epsilon,
    )

    plot_weight_mass_ratio_hist(samples, output_dir)
    plot_length_vs_weight(samples, output_dir)
    plot_trace_heatmaps(samples, output_dir, args.n_traces, args.seed)
    plot_entropy_alpha_binned(samples, output_dir)
    correlations = compute_correlations(samples, records, args.tokenizer)
    (output_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))

    ratios = [sample["weight_mass_ratio"] for sample in samples]
    print(f"{len(samples)} samples")
    print(f"weight_mass_ratio: mean={np.mean(ratios):.6f} min={min(ratios):.6f} max={max(ratios):.6f}")
    print(f"entropy_vs_length r={correlations['entropy_vs_length']} "
          f"(per-trajectory mean={correlations['entropy_vs_length_per_trajectory']['mean']})")
    stable_corr = correlations["entropy_vs_stable_weight_per_trajectory"]
    print(f"entropy_vs_stable_weight (Spearman, per-trajectory) mean={stable_corr['mean']} "
          f"median={stable_corr['median']} n={stable_corr['n_trajectories']}")
    print(f"figures + correlations.json -> {output_dir}")


if __name__ == "__main__":
    main()
