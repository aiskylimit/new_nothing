from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .scoring import DEFAULT_METRICS, VALID_METRICS, aggregate_scores, read_records, write_jsonl

DATASET_GRAPHS = {
    "cypherbench": (
        "company",
        "fictional_character",
        "flight_accident",
        "geography",
        "movie",
        "nba",
        "politics",
    ),
    "mind_the_query": ("bloom50", "healthcare", "wwc"),
    "neo4j_text2cypher": (
        "bluesky",
        "buzzoverflow",
        "companies",
        "fincen",
        "gameofthrones",
        "grandstack",
        "movies",
        "neoflix",
        "network",
        "northwind",
        "offshoreleaks",
        "recommendations",
        "stackoverflow2",
        "twitch",
        "twitter",
    ),
}


def _normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", record.get("cypher_metrics"))
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Record {record.get('id', '<unknown>')!r} has no metrics")
    for name, value in metrics.items():
        is_error = isinstance(value, dict) and isinstance(value.get("error"), str)
        if not isinstance(value, int | float) and not is_error:
            raise ValueError(
                f"Metric {name!r} for record {record.get('id', '<unknown>')!r} "
                "must be numeric or an error object"
            )
    normalized = {**record, "metrics": metrics}
    normalized.pop("cypher_metrics", None)
    return normalized


def infer_expected_graphs(input_dir: Path) -> tuple[str, ...] | None:
    return DATASET_GRAPHS.get(input_dir.name.lower())


def merge_graph_evaluations(
    input_dir: Path,
    *,
    expected_graphs: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_names = tuple(metrics)
    unknown_metrics = sorted(set(metric_names).difference(VALID_METRICS))
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {', '.join(unknown_metrics)}")
    if not metric_names:
        raise ValueError("At least one metric is required")

    expected = tuple(expected_graphs) if expected_graphs is not None else infer_expected_graphs(input_dir)
    # Ignore empty stray artifacts (for example a failed run against a wrongly
    # dotted database name) while still treating an expected empty graph as missing.
    graph_files = {
        path.parent.name: path
        for path in input_dir.glob("*/cypher_scores.jsonl")
        if path.stat().st_size > 0
    }
    if not graph_files:
        raise FileNotFoundError(f"No <graph>/cypher_scores.jsonl files found under {input_dir}")
    if expected is not None:
        missing = sorted(set(expected) - graph_files.keys())
        unexpected = sorted(graph_files.keys() - set(expected))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing graphs: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected graphs: {', '.join(unexpected)}")
            raise ValueError("Graph evaluation set is incomplete: " + "; ".join(details))

    merged: list[dict[str, Any]] = []
    per_graph: dict[str, Any] = {}
    seen_ids: set[str] = set()
    graph_order: Iterable[str] = expected if expected is not None else sorted(graph_files)
    for graph in graph_order:
        rows = [_normalized_record(row) for row in read_records(graph_files[graph])]
        if not rows:
            raise ValueError(f"Graph {graph!r} has an empty cypher_scores.jsonl")
        for row in rows:
            if row.get("graph") != graph:
                raise ValueError(
                    f"Record {row.get('id', '<unknown>')!r} in folder {graph!r} has graph={row.get('graph')!r}"
                )
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"A record in graph {graph!r} has no string id")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate evaluation record id: {record_id}")
            seen_ids.add(record_id)
        merged.extend(rows)
        per_graph[graph] = {"count": len(rows), **aggregate_scores(rows, metrics=metric_names)}

    overall = aggregate_scores(merged, metrics=metric_names)
    summary = {
        "count": len(merged),
        "graphs": per_graph,
        "overall": overall["overall"],
    }
    return merged, summary


def comma_separated(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-graph Cypher evaluation results")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dataset evaluation directory containing <graph>/cypher_scores.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --input-dir",
    )
    parser.add_argument(
        "--expected-graphs",
        type=comma_separated,
        default=None,
        help="Optional comma-separated graph list; inferred from the dataset folder name by default",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=VALID_METRICS,
        default=list(DEFAULT_METRICS),
        help="Metrics to aggregate; must match the metrics selected during graph evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    merged, summary = merge_graph_evaluations(
        args.input_dir,
        expected_graphs=args.expected_graphs,
        metrics=args.metrics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "all_graphs_cypher_scores.jsonl"
    summary_path = output_dir / "all_graphs_summary.json"
    write_jsonl(scores_path, merged)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scores": str(scores_path),
                "summary": str(summary_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
