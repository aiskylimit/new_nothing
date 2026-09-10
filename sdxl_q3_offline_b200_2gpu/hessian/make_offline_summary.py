#!/usr/bin/env python3
"""Create the one small artifact users copy back from an offline run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args()

    score_path = args.eval_dir / "scores" / "full_scores.csv"
    if not score_path.is_file():
        raise FileNotFoundError(f"Evaluation score table missing: {score_path}")
    with score_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = ("pickscore", "hpsv2", "aesthetics", "clip", "imagereward")
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        key = f"{row['dataset']}/{row['model']}"
        bucket = grouped.setdefault(key, {metric: [] for metric in metrics})
        for metric in metrics:
            if row.get(metric) not in (None, ""):
                bucket[metric].append(float(row[metric]))
    means = {
        key: {metric: statistics.fmean(values) for metric, values in values_by_metric.items() if values}
        for key, values_by_metric in grouped.items()
    }
    datasets = sorted({row["dataset"] for row in rows})
    payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "sdxl",
        "method": "q3_dspo",
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_path": str(args.run_dir),
        "evaluation_score_table": str(score_path),
        "datasets": datasets,
        "samples_scored": len(rows),
        "seed": args.seed,
        "config": {
            "mode": args.mode,
            "num_gpus": args.num_gpus,
            "effective_batch": args.effective_batch,
            "sampling_steps": 50,
            "cfg": 7.5,
            "resolution": 1024,
        },
        "mean_metrics": means,
        "logs": str(args.run_dir.parent.parent / "logs"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    size = args.output.stat().st_size
    if size >= 25 * 1024 * 1024:
        raise RuntimeError(f"Summary exceeds 25 MiB: {size} bytes")
    print(json.dumps({"output": str(args.output), "bytes": size, "rows": len(rows)}))


if __name__ == "__main__":
    main()

