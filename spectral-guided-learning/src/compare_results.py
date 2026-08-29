"""Phase 7 helper: merge per-model eval summaries into the final comparison table.

    python src/compare_results.py [--results-dir results]
"""

import argparse
import json
from pathlib import Path

BENCHMARK_ORDER = ["aime24", "aime25", "math500", "amc12"]
# math500 is primary: AIME (30 problems) and AMC12 (83 problems) are too small to be reliable alone.
PRIMARY_BENCHMARKS = {"math500"}


def load_summaries(results_dir: Path) -> dict[str, dict[str, float]]:
    """{model_tag: {benchmark: accuracy}} from results/<tag>/summary.json files."""
    models = {}
    for path in sorted(results_dir.glob("*/summary.json")):
        rows = json.loads(path.read_text())
        tag = rows[0]["model"] if rows else path.parent.name
        models[tag] = {row["benchmark"]: row["accuracy"] for row in rows}
    return models


METHODS = ("vanilla", "spectral", "prucot")


def split_tag(tag: str) -> tuple[str, str] | None:
    """"spectral-qwen3-1.7b" -> ("spectral", "qwen3-1.7b"); None if no known method prefix."""
    for method in METHODS:
        prefix = f"{method}-"
        if tag.startswith(prefix):
            return method, tag[len(prefix) :]
    return None


def format_table(models: dict[str, dict[str, float]]) -> str:
    benchmarks = [name for name in BENCHMARK_ORDER if any(name in row for row in models.values())]
    header = "| Model | " + " | ".join(benchmarks) + " | Avg | Primary avg |"
    separator = "|" + "---|" * (len(benchmarks) + 3)

    lines = [header, separator]
    for tag, scores in models.items():
        values = [scores.get(name) for name in benchmarks]
        present = [value for value in values if value is not None]
        primary = [scores[name] for name in benchmarks if name in PRIMARY_BENCHMARKS and name in scores]
        cells = [f"{value:.1%}" if value is not None else "-" for value in values]
        lines.append(
            f"| {tag} | " + " | ".join(cells) + f" | {sum(present) / len(present):.1%} | "
            + (f"{sum(primary) / len(primary):.1%}" if primary else "-") + " |"
        )

    # Group by track so vanilla-vs-spectral/prucot deltas are computed per model line
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
        for method in ("spectral", "prucot"):
            if method not in by_method:
                continue
            new_scores = models[by_method[method]]
            deltas = []
            for name in benchmarks:
                base, new = base_scores.get(name), new_scores.get(name)
                deltas.append(f"{(new - base) * 100:+.1f}" if base is not None and new is not None else "-")
            lines.append(f"| **delta ({method} - vanilla, {track}) (pp)** | " + " | ".join(deltas) + " | | |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models = load_summaries(results_dir)
    if not models:
        raise SystemExit(f"no */summary.json found in {results_dir}")

    table = format_table(models)
    print(table)
    (results_dir / "comparison-table.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
