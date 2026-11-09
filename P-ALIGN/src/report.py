#!/usr/bin/env python3
"""Write pass@1 / pass@3 per benchmark and unweighted average to a txt file."""
import argparse
import json
from pathlib import Path

BENCH = [
    ("AIME25", "output/result/aime25_scored.jsonl"),
    ("AIME24", "output/result/aime24_scored.jsonl"),
    ("AMC12", "output/result/amc12_scored.jsonl"),
    ("MATH500", "output/result/math500_scored.jsonl"),
]


def metrics(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0
    # Pass@1 = mean accuracy over the k samples per problem
    p1 = 0.0
    # Pass@3 = 1 if any of the k samples is correct
    p3 = 0
    for r in rows:
        labels = [int(x) for x in (r.get("label") or [])]
        if labels:
            p1 += sum(labels) / len(labels)
        if r.get("passn") or any(labels):
            p3 += 1
    return 100.0 * p1 / n, 100.0 * p3 / n, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/eval_results.txt")
    args = p.parse_args()
    lines = []
    p1s, p3s = [], []
    for name, rel in BENCH:
        path = Path(rel)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        p1, p3, n = metrics(path)
        p1s.append(p1)
        p3s.append(p3)
        lines.append(f"{name:8s}  n={n:3d}  pass@1={p1:.2f}  pass@3={p3:.2f}")
    lines.append(
        f"{'Avg':8s}           pass@1={sum(p1s) / len(p1s):.2f}  pass@3={sum(p3s) / len(p3s):.2f}"
    )
    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
