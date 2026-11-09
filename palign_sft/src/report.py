#!/usr/bin/env python3
"""Write pass@1/pass@3 per benchmark and their unweighted average."""

import argparse
import json
from pathlib import Path

BENCHMARKS = [
    ("AIME25", "aime25_scored.jsonl", 30),
    ("AIME24", "aime24_scored.jsonl", 30),
    ("AMC12", "amc12_scored.jsonl", 83),
    ("MATH500", "math500_scored.jsonl", 500),
]


def metrics(path: Path, expected_n: int, expected_rows: int):
    if expected_n <= 0:
        raise ValueError("expected_n must be positive")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = len(rows)
    if count != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {count}")
    pass_one = 0.0
    pass_n = 0
    for index, row in enumerate(rows):
        labels = [int(value) for value in (row.get("label") or [])]
        if len(labels) != expected_n:
            raise ValueError(
                f"{path}: row {index} expected {expected_n} labels, found {len(labels)}"
            )
        pass_one += sum(labels) / len(labels)
        if any(labels):
            pass_n += 1
    return 100.0 * pass_one / count, 100.0 * pass_n / count, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=3)
    args = parser.parse_args()
    lines = []
    pass_ones = []
    pass_ns = []
    for name, filename, expected_rows in BENCHMARKS:
        path = args.result_dir / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        pass_one, pass_n, count = metrics(path, args.expected_n, expected_rows)
        pass_ones.append(pass_one)
        pass_ns.append(pass_n)
        lines.append(
            f"{name:8s}  n={count:3d}  pass@1={pass_one:.2f}  "
            f"pass@{args.expected_n}={pass_n:.2f}"
        )
    lines.append(
        f"{'Avg':8s}           pass@1={sum(pass_ones) / len(pass_ones):.2f}  "
        f"pass@{args.expected_n}={sum(pass_ns) / len(pass_ns):.2f}"
    )
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
