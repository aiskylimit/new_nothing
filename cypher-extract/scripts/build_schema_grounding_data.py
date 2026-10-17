#!/usr/bin/env python3
"""Build normalized schema-grounding supervision from local Text-to-Cypher benchmarks."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from schema_grounding.datasets import SUPPORTED_SOURCES  # noqa: E402
from schema_grounding.pipeline import build_dataset  # noqa: E402


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks",
        help="Directory containing Cypherbench, Mind_the_query, and Neo4j_Text2Cypher.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "schema_grounding",
        help="Directory for generated JSONL files (refuses to overwrite by default).",
    )
    parser.add_argument(
        "--sources",
        type=_csv,
        default=SUPPORTED_SOURCES,
        help="Comma-separated sources: cypherbench,mind_the_query,neo4j_text2cypher.",
    )
    parser.add_argument(
        "--splits",
        type=_csv,
        default=("train",),
        help="Comma-separated structured benchmark splits: train, dev, test. Default: train.",
    )
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="Keep examples with unmapped or unmatched Cypher patterns; strict coverage is default.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=None,
        help="Optional maximum count of negative units per positive unit. Default keeps all units.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap per source/split, useful for inspection and smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.with_name(f"{output_dir.name}_{timestamp}")
        print(f"Output directory not empty. Redirecting output to: {output_dir}", file=sys.stderr)

    manifest = build_dataset(
        benchmarks_root=args.benchmarks_root,
        output_dir=output_dir,
        sources=args.sources,
        splits=args.splits,
        strict=not args.allow_partial_coverage,
        negative_ratio=args.negative_ratio,
        max_examples=args.max_examples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
