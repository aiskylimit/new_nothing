#!/usr/bin/env python3
"""Discover and evaluate the latest valid checkpoint from every selected run."""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

from eval_utils import (atomic_json, classify_checkpoint, output_dir, run_identity_name,
                        select_checkpoint, suite_config, utc_now)


def is_completed_summary(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("overall_status") == "complete"
    except Exception:
        return False


def is_mode_complete(path, mode):
    try:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = summary.get("benchmarks") or []
        if not rows:
            return False
        if mode == "infer":
            return all(row.get("inference_status") == "complete" for row in rows)
        if mode == "eval":
            return all(row.get("scoring_status") == "complete" for row in rows)
        return summary.get("overall_status") == "complete"
    except Exception:
        return False


def discover_runs(root):
    """Discover artifact-backed runs at any depth, excluding evaluation outputs."""
    runs = set()
    for metadata in list(root.rglob("adapter_config.json")) + list(root.rglob("config.json")):
        directory = metadata.parent
        try:
            relative = directory.relative_to(root)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] == "eval":
            continue
        runs.add(directory.parent if directory.name.startswith("checkpoint-") else directory)
    return sorted(runs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--python-bin", type=Path, required=True)
    p.add_argument("--outputs-root", type=Path, required=True)
    p.add_argument("--suite", default="requested_benchmarks")
    p.add_argument("--pattern", default="*")
    p.add_argument("--mode", choices=("all", "infer", "eval"), default="all")
    p.add_argument("--attention-backend", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--merge-device-map", default="cpu")
    p.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--base-model-override", action="append", default=[])
    args = p.parse_args()
    project = args.project_dir.resolve()
    root = args.outputs_root.resolve()
    cfg = suite_config(project / "configs/eval" / f"{args.suite}.json")
    candidates = []
    discovery_failures = 0
    records = []
    for run in discover_runs(root) if root.is_dir() else []:
        relative_name = str(run.relative_to(root))
        if not (fnmatch.fnmatch(run.name, args.pattern) or fnmatch.fnmatch(relative_name, args.pattern)):
            continue
        checkpoint = select_checkpoint(run)
        if checkpoint:
            candidates.append((run, checkpoint))
        else:
            kind, reasons = classify_checkpoint(run)
            print(f"[{kind}] {run}: no complete checkpoint ({', '.join(reasons)})", file=sys.stderr)
            discovery_failures += 1
            records.append({"run": str(run), "checkpoint": None, "status": "failed",
                            "checkpoint_kind": kind, "error": ", ".join(reasons)})
    if not candidates:
        if records and not args.dry_run:
            atomic_json(root / "eval" / cfg.get("name", args.suite) / "batch_summary.json",
                        {"suite": cfg.get("name", args.suite), "updated_at": utc_now(),
                         "pattern": args.pattern, "records": records})
        print(f"No valid checkpoints found under {root} matching {args.pattern!r}", file=sys.stderr)
        return 1
    failed, passed, skipped = discovery_failures, 0, 0
    batch_path = root / "eval" / cfg.get("name", args.suite) / "batch_summary.json"

    def save_batch():
        if not args.dry_run:
            atomic_json(batch_path, {"suite": cfg.get("name", args.suite), "updated_at": utc_now(),
                                     "pattern": args.pattern, "records": records})

    save_batch()
    for run, checkpoint in candidates:
        identity = run_identity_name(checkpoint, root)
        target = output_dir(root, cfg.get("name", args.suite), identity, checkpoint)
        summary = target / "summary.json"
        if args.reuse and is_mode_complete(summary, args.mode):
            print(f"[done] {run.name}: {checkpoint.name}")
            skipped += 1
            records.append({"run": str(run), "checkpoint": str(checkpoint),
                            "status": "complete" if args.mode == "all" else f"{args.mode}-complete",
                            "summary": str(summary), "skip_reason": "canonical summary complete"})
            save_batch()
            continue
        cmd = [str(project / "scripts/eval/run_eval.sh"), "--checkpoint", str(checkpoint),
               "--suite", args.suite, "--mode", args.mode, "--outputs-root", str(root),
               "--attention-backend", args.attention_backend,
               "--max-new-tokens", str(args.max_new_tokens),
               "--device-map", args.device_map,
               "--merge-device-map", args.merge_device_map]
        if args.dry_run: cmd.append("--dry-run")
        if not args.reuse: cmd.append("--no-reuse")
        for item in args.base_model_override: cmd += ["--base-model-override", item]
        print(f"[run] {run.name}: {checkpoint.name} ({classify_checkpoint(checkpoint)[0]})")
        result = subprocess.run(cmd, cwd=project, env=os.environ.copy())
        mode_complete = args.dry_run or is_mode_complete(target / "summary.json", args.mode)
        if result.returncode or not mode_complete:
            failed += 1
            print(f"[failed] {run.name}", file=sys.stderr)
            records.append({"run": str(run), "checkpoint": str(checkpoint), "status": "failed",
                            "checkpoint_kind": classify_checkpoint(checkpoint)[0],
                            "returncode": result.returncode, "summary": str(target / "summary.json"),
                            "error": "launcher failed" if result.returncode else f"mode {args.mode} incomplete"})
        else:
            passed += 1
            records.append({"run": str(run), "checkpoint": str(checkpoint),
                            "status": ("dry-run" if args.dry_run else
                                       ("complete" if args.mode == "all" else f"{args.mode}-complete")),
                            "checkpoint_kind": classify_checkpoint(checkpoint)[0],
                            "summary": str(target / "summary.json")})
        save_batch()
    print(f"Summary: passed={passed} failed={failed} completed-skips={skipped}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
