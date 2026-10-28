#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SELECTED_PAPER_BENCHMARKS = ("aime25", "hmmt25")
ACTIVE_BENCHMARKS = ("aime24", "aime25", "aime26", "hmmt25")


def load_rows(results_root: Path, metric: str = "average", benchmarks=ACTIVE_BENCHMARKS):
    rows = []
    for path in sorted(results_root.glob("*/*/*/*.json")):
        relative = path.relative_to(results_root)
        model, method, step, filename = relative.parts
        dataset = Path(filename).stem
        if dataset not in benchmarks:
            continue
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        row = {
            "model": model,
            "method": method,
            "step": step,
            "dataset": dataset,
            "val_n": result.get("val_n"),
        }
        if metric == "pass":
            row["pass_at_n"] = result["pass_at_n_pct"]
        else:
            row.update({
                "average_at_n": result["average_at_n_pct"],
                "pass_at_n": result["pass_at_n_pct"],
                "majority_vote_at_n": result["majority_vote_at_n_pct"],
                "format_rate": result["format_rate"],
            })
        row["result_file"] = str(path)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_results_table(rows: list[dict], metric: str = "average", benchmarks=ACTIVE_BENCHMARKS) -> str:
    score_key = "pass_at_n" if metric == "pass" else "average_at_n"
    label = "Pass@12" if metric == "pass" else "Avg@12"
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["method"], row["step"])][row["dataset"]] = row[score_key]

    def step_key(item):
        model, method, step = item[0]
        return model, method, -1 if step == "base" else int(step)

    header = "| Model | Method | Step | " + " | ".join(
        f"{name.upper()} {label}" for name in benchmarks
    )
    if metric != "pass":
        header += " | Average"
    lines = [header + " |", "|---|---|---:|" + "---:|" * (len(benchmarks) + (metric != "pass"))]
    for (model, method, step), scores in sorted(grouped.items(), key=step_key):
        values = [scores.get(dataset) for dataset in benchmarks]
        formatted = ["-" if value is None else f"{value:.1f}" for value in values]
        scores_text = f"{' | '.join(formatted)}"
        if metric != "pass":
            average = sum(value for value in values if value is not None) / sum(value is not None for value in values)
            scores_text += f" | {average:.1f}"
        lines.append(f"| {model} | {method} | {step} | {scores_text} |")
    return "\n".join(lines)


def select_best(rows: list[dict]):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["method"], row["step"])][row["dataset"]] = row

    candidates = []
    for (model, method, step), by_dataset in grouped.items():
        if not all(dataset in by_dataset for dataset in SELECTED_PAPER_BENCHMARKS):
            continue
        candidate = {"model": model, "method": method, "step": step}
        candidate.update({dataset: None for dataset in ACTIVE_BENCHMARKS})
        paper_scores = []
        for dataset in ACTIVE_BENCHMARKS:
            if dataset not in by_dataset:
                continue
            score = by_dataset[dataset]["average_at_n"]
            candidate[dataset] = score
            if dataset in SELECTED_PAPER_BENCHMARKS:
                paper_scores.append(score)
        candidate["selected_paper_average"] = sum(paper_scores) / len(paper_scores)
        if all(dataset in by_dataset for dataset in ACTIVE_BENCHMARKS):
            candidate["extended_average"] = sum(
                by_dataset[dataset]["average_at_n"] for dataset in ACTIVE_BENCHMARKS
            ) / len(ACTIVE_BENCHMARKS)
        else:
            candidate["extended_average"] = None
        candidates.append(candidate)

    best = {}
    for candidate in candidates:
        key = (candidate["model"], candidate["method"])
        if key not in best or candidate["selected_paper_average"] > best[key]["selected_paper_average"]:
            best[key] = candidate
    return sorted(best.values(), key=lambda row: (row["model"], row["method"]))


def main():
    parser = argparse.ArgumentParser(description="Aggregate OPSD paper benchmark JSON files.")
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metric", choices=("average", "pass"), default="average")
    parser.add_argument("--datasets", default=" ".join(ACTIVE_BENCHMARKS))
    args = parser.parse_args()
    benchmarks = tuple(args.datasets.split())
    if not benchmarks or any(dataset not in ACTIVE_BENCHMARKS for dataset in benchmarks):
        parser.error("--datasets must contain space-separated supported benchmark names")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_root, metric=args.metric, benchmarks=benchmarks)
    if not rows:
        raise FileNotFoundError(f"No evaluation JSON files found under {args.results_root}")

    write_csv(args.output_dir / "all_results.csv", rows)
    if args.metric == "average":
        best = select_best(rows)
        write_csv(args.output_dir / "best_results.csv", best)
        with (args.output_dir / "best_results.json").open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)

    table = render_results_table(rows, metric=args.metric, benchmarks=benchmarks)
    (args.output_dir / "results_table.md").write_text(f"{table}\n", encoding="utf-8")
    print(table)

    print(f"Wrote {len(rows)} result rows to {args.output_dir / 'all_results.csv'}")
    if args.metric == "average":
        print(f"Wrote {len(best)} best model/method rows to {args.output_dir / 'best_results.csv'}")
    print(f"Wrote result table to {args.output_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()
