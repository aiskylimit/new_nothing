#!/usr/bin/env python3
"""Collect canonical evaluation summaries into human- and machine-readable reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_DIR / "outputs/eval/requested_benchmarks"
VLMEVALKIT_DIR = PROJECT_DIR / "VLMEvalKit"

CSV_FIELDS = (
    "summary_path",
    "run_name",
    "model_name",
    "checkpoint",
    "checkpoint_kind",
    "architecture",
    "suite",
    "overall_status",
    "dataset",
    "inference_status",
    "scoring_status",
    "primary_metric_name",
    "primary_metric_value",
    "error",
    "skip_reason",
    "native_result_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory recursively searched for summary.json (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        default=[],
        help="Read an explicit summary.json; repeat to select multiple files instead of --root.",
    )
    parser.add_argument("--csv", type=Path, help="Combined long-form CSV output path.")
    parser.add_argument("--json", type=Path, help="Combined JSON output path.")
    parser.add_argument("--no-write", action="store_true", help="Only print the report to stdout.")
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit non-zero if any selected summary is not complete.",
    )
    return parser.parse_args()


def json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def is_missing_metric(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return not value or all(is_missing_metric(item) for item in value.values())
    return False


def latest_native_status(summary_path: Path) -> Path | None:
    native_root = summary_path.parent / "native"
    if not native_root.is_dir():
        return None
    candidates = []
    for candidate in native_root.rglob("status.json"):
        try:
            candidates.append((candidate.stat().st_mtime, candidate))
        except OSError:
            continue
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def reporter_primary_metric(dataset: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Reproduce VLMEvalKit's display-metric selection without running evaluation."""
    if not VLMEVALKIT_DIR.is_dir():
        return {}
    local_lmu_data = PROJECT_DIR / "eval_data/LMUData"
    if "LMUData" not in os.environ and local_lmu_data.is_dir():
        os.environ["LMUData"] = str(local_lmu_data)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    kit_path = str(VLMEVALKIT_DIR)
    if kit_path not in sys.path:
        sys.path.insert(0, kit_path)
    try:
        from vlmeval.smp.status_report import _resolve_dataset_reporter

        reporter = _resolve_dataset_reporter(dataset)
        resolved = reporter.report_primary_metric(metrics)
    except Exception:
        return {}
    return resolved if isinstance(resolved, dict) else {}


def known_display_metric(name: Any, metrics: dict[str, Any]) -> Any:
    """Resolve stable label/key conversions used by the requested MCQ datasets."""
    source_keys = {
        "Overall Acc": ("split=none|Overall",),
        "Overall Acc (val)": ("split=validation|Overall", "split=val|Overall"),
        "Overall Acc (test)": ("split=test|Overall",),
    }
    for key in source_keys.get(name, ()):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value * 100
    return None


def metric_from_status(dataset: str, benchmark: dict[str, Any], status: dict[str, Any]) -> tuple[Any, Any]:
    name = benchmark.get("primary_metric_name") or status.get("primary_metric")
    stored_value = status.get("primary_metric_value")
    if not is_missing_metric(stored_value):
        return name, stored_value

    metrics = status.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return name, None

    if isinstance(name, list):
        direct = {key: metrics.get(key) for key in name}
        if not is_missing_metric(direct):
            return name, direct
    elif name in metrics:
        return name, metrics[name]

    known_value = known_display_metric(name, metrics)
    if not is_missing_metric(known_value):
        return name, known_value

    resolved = reporter_primary_metric(dataset, metrics)
    if not resolved:
        return name, None
    if isinstance(name, list):
        selected = {key: resolved.get(key) for key in name}
        if not is_missing_metric(selected):
            return name, selected
    elif name in resolved:
        return name, resolved[name]
    if len(resolved) == 1:
        resolved_name, resolved_value = next(iter(resolved.items()))
        return name or resolved_name, resolved_value
    return name, None


def backfill_metrics_from_status(summary: dict[str, Any], summary_path: Path) -> None:
    missing = [
        item for item in summary["benchmarks"]
        if isinstance(item, dict) and is_missing_metric(item.get("primary_metric_value"))
    ]
    if not missing:
        return
    status_path = latest_native_status(summary_path)
    if status_path is None:
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    datasets = status.get("datasets") if isinstance(status, dict) else None
    if not isinstance(datasets, dict):
        return

    for benchmark in missing:
        dataset = benchmark.get("dataset")
        native = datasets.get(dataset)
        if not isinstance(dataset, str) or not isinstance(native, dict):
            continue
        name, value = metric_from_status(dataset, benchmark, native)
        if is_missing_metric(value):
            continue
        benchmark["primary_metric_name"] = name
        benchmark["primary_metric_value"] = value


def read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Summary must contain a JSON object: {path}")
    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError(f"Summary has no valid benchmarks list: {path}")
    value["summary_path"] = str(path.resolve())
    backfill_metrics_from_status(value, path.resolve())
    return value


def flatten(summary: dict[str, Any]) -> list[dict[str, str]]:
    common = {
        "summary_path": summary.get("summary_path"),
        "run_name": summary.get("run_name"),
        "model_name": summary.get("model_name"),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_kind": summary.get("checkpoint_kind"),
        "architecture": summary.get("architecture"),
        "suite": summary.get("suite"),
        "overall_status": summary.get("overall_status"),
    }
    rows = []
    for benchmark in summary["benchmarks"]:
        if not isinstance(benchmark, dict):
            continue
        row = {
            **common,
            "dataset": benchmark.get("dataset"),
            "inference_status": benchmark.get("inference_status"),
            "scoring_status": benchmark.get("scoring_status"),
            "primary_metric_name": benchmark.get("primary_metric_name"),
            "primary_metric_value": benchmark.get("primary_metric_value"),
            "error": benchmark.get("error"),
            "skip_reason": benchmark.get("skip_reason"),
            "native_result_path": benchmark.get("native_result_path"),
        }
        rows.append({key: json_value(row.get(key)) for key in CSV_FIELDS})
    return rows


def clipped(value: Any, width: int) -> str:
    text = json_value(value).replace("\n", " ") or "-"
    return text if len(text) <= width else text[: width - 3] + "..."


def print_report(summaries: list[dict[str, Any]]) -> None:
    print("Evaluation summary")
    print("=" * 80)
    for summary in summaries:
        benchmarks = [item for item in summary["benchmarks"] if isinstance(item, dict)]
        completed = sum(
            item.get("inference_status") == "complete" and item.get("scoring_status") == "complete"
            for item in benchmarks
        )
        checkpoint = Path(str(summary.get("checkpoint", "unknown"))).name
        print(
            f"\nRun: {summary.get('run_name', '-')} | checkpoint: {checkpoint} | "
            f"status: {summary.get('overall_status', '-')} | complete: {completed}/{len(benchmarks)}"
        )
        print(f"{'dataset':<27} {'infer':<9} {'score':<9} {'metric':<22} {'value':<14} error")
        print("-" * 110)
        for item in benchmarks:
            print(
                f"{clipped(item.get('dataset'), 27):<27} "
                f"{clipped(item.get('inference_status'), 9):<9} "
                f"{clipped(item.get('scoring_status'), 9):<9} "
                f"{clipped(item.get('primary_metric_name'), 22):<22} "
                f"{clipped(item.get('primary_metric_value'), 14):<14} "
                f"{clipped(item.get('error') or item.get('skip_reason'), 80)}"
            )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, root: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(str(item.get("overall_status", "unknown")) for item in summaries)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "total_summaries": len(summaries),
        "status_counts": dict(sorted(status_counts.items())),
        "summaries": summaries,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    paths = [path.expanduser().resolve() for path in args.summary]
    if not paths:
        paths = sorted(root.rglob("summary.json")) if root.is_dir() else []
    if not paths:
        print(f"ERROR: no summary.json found under {root}", file=sys.stderr)
        return 2

    summaries = []
    errors = []
    for path in paths:
        try:
            summaries.append(read_summary(path))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    summaries.sort(key=lambda item: (str(item.get("run_name", "")), str(item.get("checkpoint", ""))))
    rows = [row for summary in summaries for row in flatten(summary)]
    print_report(summaries)

    if not args.no_write:
        csv_path = (args.csv or root / "combined_summary.csv").expanduser().resolve()
        json_path = (args.json or root / "combined_summary.json").expanduser().resolve()
        write_csv(csv_path, rows)
        write_json(json_path, root, summaries)
        print(f"\nCSV:  {csv_path}")
        print(f"JSON: {json_path}")

    incomplete = [item for item in summaries if item.get("overall_status") != "complete"]
    return 1 if args.fail_on_incomplete and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
