"""Phase 7 helper: merge per-model eval summaries into the final comparison table.

    python sgl.eval.compare [--results-dir results]
"""

import argparse
import json
from pathlib import Path

BENCHMARK_ORDER = ["aime24", "aime25", "math500", "amc12"]
# math500 is primary: AIME (30 problems) and AMC12 (83 problems) are too small to be reliable alone.
PRIMARY_BENCHMARKS = {"math500"}


def load_summaries(results_dir: Path) -> tuple[dict[str, dict[str, dict[str, float]]], int]:
    """{model_tag: {benchmark: {metric: value}}} plus the largest k seen across summaries.

    k is the number of samples drawn per problem at eval time, so a run evaluated with
    --n-samples 3 carries a "pass@3" key alongside "pass@1".
    """
    models: dict[str, dict[str, dict[str, float]]] = {}
    samples = 1
    for path in sorted(results_dir.glob("*/summary.json")):
        rows = json.loads(path.read_text())
        tag = rows[0]["model"] if rows else path.parent.name
        models[tag] = {row["benchmark"]: row for row in rows}
        for row in rows:
            samples = max(samples, row.get("samples_per_problem", 1))
    return models, samples


# Model tracks (scripts/*/<phase>_<track>.sh); a tag is "<method>-<track>", where method may itself
# carry dashes ("iwc-stable", "trans-vanilla-l0.3"), so the split is anchored on the track suffix.
TRACKS = ("qwen3-8b", "qwen25-7b", "r1-qwen-1.5b", "r1-qwen-7b")


def split_tag(tag: str) -> tuple[str, str] | None:
    """"spectral-r1-qwen-1.5b" -> ("spectral", "r1-qwen-1.5b"); None if no known track suffix."""
    for track in TRACKS:
        suffix = f"-{track}"
        if tag.endswith(suffix) and len(tag) > len(suffix):
            return tag[: -len(suffix)], track
    return None


def format_table(models: dict[str, dict[str, dict[str, float]]], metric: str) -> str:
    """One table for one metric ("pass@1" / "pass@3"); rows are model tags, plus per-track deltas."""
    def score(scores: dict[str, dict[str, float]], name: str) -> float | None:
        row = scores.get(name)
        if row is None:
            return None
        # "accuracy" is the pre-pass@k key and always equals pass@1.
        return row.get(metric, row.get("accuracy") if metric == "pass@1" else None)

    benchmarks = [name for name in BENCHMARK_ORDER if any(name in row for row in models.values())]
    header = "| Model | " + " | ".join(benchmarks) + " | Avg | Primary avg |"
    separator = "|" + "---|" * (len(benchmarks) + 3)

    lines = [header, separator]
    for tag, scores in models.items():
        values = [score(scores, name) for name in benchmarks]
        present = [value for value in values if value is not None]
        primary = [
            value
            for name in benchmarks
            if name in PRIMARY_BENCHMARKS and (value := score(scores, name)) is not None
        ]
        cells = [f"{value:.2%}" if value is not None else "-" for value in values]
        if not present:
            continue
        lines.append(
            f"| {tag} | " + " | ".join(cells) + f" | {sum(present) / len(present):.2%} | "
            + (f"{sum(primary) / len(primary):.2%}" if primary else "-") + " |"
        )

    # Group by track so the vanilla-vs-spectral delta is computed per model line
    # (tags are always "<method>-<track>", never bare "vanilla"/"spectral").
    tracks: dict[str, dict[str, str]] = {}
    for tag in models:
        parsed = split_tag(tag)
        if parsed:
            method, track = parsed
            tracks.setdefault(track, {})[method] = tag

    for track, by_method in tracks.items():
        if "vanilla" not in by_method:
            continue
        base_scores = models[by_method["vanilla"]]
        for method in by_method:
            if method == "vanilla":
                continue
            new_scores = models[by_method[method]]
            deltas = []
            for name in benchmarks:
                base, new = score(base_scores, name), score(new_scores, name)
                deltas.append(f"{(new - base) * 100:+.2f}" if base is not None and new is not None else "-")
            lines.append(f"| **delta ({method} - vanilla, {track}) (pp)** | " + " | ".join(deltas) + " | | |")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    models, samples = load_summaries(results_dir)
    if not models:
        raise SystemExit(f"no */summary.json found in {results_dir}")

    # P-ALIGN reports Pass@1 and Pass@3 side by side; emit one table per metric the eval
    # runs actually produced (a k=1 run only has Pass@1).
    metrics = ["pass@1"] + ([f"pass@{samples}"] if samples > 1 else [])
    table = "\n\n".join(
        f"### {metric.replace('pass@', 'Pass@')}\n\n" + format_table(models, metric)
        for metric in metrics
    )
    print(table)
    (results_dir / "comparison-table.md").write_text(table + "\n")

    summary_path = results_dir / "eval-summary.json"
    summary_path.write_text(json.dumps(models, indent=2))
    print(f"per-model eval summary -> {summary_path}")


if __name__ == "__main__":
    main()
