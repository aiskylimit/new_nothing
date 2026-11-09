"""Shared rebuttal evaluation: P-ALIGN prompts, vLLM, and math-verify."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
BENCHMARKS = ("aime25", "aime24", "amc12", "math500")


def summarize(rows, n):
    if not rows:
        raise ValueError("evaluation set is empty")
    counts = []
    for index, row in enumerate(rows):
        labels = row.get("label", [])
        if len(labels) != n or any(x not in (0, 1) for x in labels):
            raise ValueError(f"row {index}: expected {n} binary labels")
        counts.append(sum(labels))
    result = {
        "num_problems": len(rows),
        "num_generations": n,
        "pass@1": 100 * sum(c / n for c in counts) / len(rows),
        f"pass@{n}": 100 * sum(c > 0 for c in counts) / len(rows),
    }
    if n >= 3:
        result["pass@3"] = 100 * sum(
            1 - math.comb(n - c, 3) / math.comb(n, 3) for c in counts
        ) / len(rows)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", "--model-name-or-path", required=True)
    p.add_argument("--data-dir", type=Path, default=SOURCE.parent / "data/raw")
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    p.add_argument("--input-files", "--input_files", nargs="+", type=Path)
    p.add_argument("--result-dir", "--output-dir", type=Path, required=True)
    p.add_argument("--n", "--num-generations", type=int, default=int(os.environ.get("EVAL_N", "3")))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-tokens", "--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--score-only", action="store_true")
    return p.parse_args(argv)


def run(args):
    if args.n < 1 or args.max_tokens < 1 or args.batch_size < 1:
        raise ValueError("n, max_tokens and batch_size must be positive")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("max_model_len must leave room for the prompt beyond max_tokens")
    inputs = args.input_files or [args.data_dir / f"{name}.jsonl" for name in args.benchmarks]
    names = [path.stem for path in inputs]
    if len(names) != len(set(names)):
        raise ValueError("input files must have distinct stems")
    counts, fingerprints = {}, {}
    for name, path in zip(names, inputs):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError(f"{path} is empty")
        counts[name] = len(records)
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.result_dir / f"{name}.jsonl" for name in names]
    if not args.score_only:
        if not args.overwrite and any(path.exists() for path in outputs):
            raise FileExistsError("raw outputs already exist; use another result-dir or --overwrite")
        config = vars(args).copy()
        config.update(backend="vllm", grader="math-verify", input_sha256=fingerprints)
        (args.result_dir / "evaluation_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n"
        )
        subprocess.run([
            sys.executable, str(SOURCE / "test.py"), "--model", args.model,
            "--input_files", *map(str, inputs), "--output_files", *map(str, outputs),
            "--n", str(args.n), "--temperature", str(args.temperature),
            "--top_p", str(args.top_p), "--repetition_penalty", str(args.repetition_penalty),
            "--max_tokens", str(args.max_tokens), "--max_model_len", str(args.max_model_len),
            "--batch_size", str(args.batch_size), "--seed", str(args.seed),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        ], check=True)
    summary = {}
    for name, raw in zip(names, outputs):
        scored = raw.with_name(f"{name}_scored.jsonl")
        subprocess.run([
            sys.executable, str(SOURCE / "evaluation.py"), "--input_path", str(raw),
            "--output_path", str(scored), "--expected_n", str(args.n),
        ], check=True)
        rows = [json.loads(line) for line in scored.read_text().splitlines() if line.strip()]
        if len(rows) != counts[name]:
            raise ValueError(f"{name}: expected {counts[name]} scored rows, got {len(rows)}")
        summary[name] = summarize(rows, args.n)
    metric_names = sorted(set.intersection(*(set(row) for row in summary.values())))
    summary["average"] = {
        key: sum(row[key] for row in summary.values()) / len(summary)
        for key in metric_names if key.startswith("pass@")
    }
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [name + "  " + "  ".join(
        f"{key}={value:.2f}" for key, value in metrics.items() if key.startswith("pass@")
    ) for name, metrics in summary.items()]
    report = "\n".join(lines) + "\n"
    (args.result_dir / "eval_results.txt").write_text(report)
    print(report, end="")
    return summary


if __name__ == "__main__":
    run(parse_args())
