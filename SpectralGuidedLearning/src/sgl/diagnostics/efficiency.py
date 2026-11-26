"""Efficiency table of a set of runs: pipeline cost, training cost, generation length, accuracy.

Reads <efficiency-dir>/<run>.jsonl (written by the stage runner, see sgl.efficiency), each run's
<checkpoint>/run-summary.json and <results_dir>/<run>/summary.json. Writes table.md and table.json
into the efficiency dir.

Per stage the latest successful execution counts. `prepare` is shared by the arms of one track and
runs only once, so an arm without its own record takes the track's. A stage with no record at all
shows as "-" and leaves the totals empty rather than too small.
"""

import argparse
import json
from pathlib import Path

# stages that turn raw traces into training data (analysis-only stages such as answer_stats are not cost)
PREPROCESSING = ("prepare", "vanilla", "capture", "answer_gain", "gain_signal", "weights", "answer_mask")


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def load_runs(directory: Path) -> list[dict]:
    runs = []
    for path in sorted(directory.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        stages = {r["stage"]: r for r in records if r["exit_code"] == 0}
        if records:
            runs.append({"run": path.stem, "latest": records[-1], "stages": stages})
    return runs


def summarize_run(run: dict, shared_prepare: dict, root: Path) -> dict:
    latest, stages = run["latest"], dict(run["stages"])
    pipeline = latest.get("pipeline") or []
    if "prepare" in pipeline and "prepare" not in stages and latest["track"] in shared_prepare:
        stages["prepare"] = shared_prepare[latest["track"]]

    seconds = {name: stages[name]["seconds"] for name in stages}
    needed = [name for name in pipeline if name in PREPROCESSING]
    preprocessing = sum(seconds[n] for n in needed) if all(n in seconds for n in needed) else None
    train = seconds.get("train") if "train" in pipeline else 0.0
    peaks = {name: r["peak_gpu_mib"] for name, r in stages.items() if r.get("peak_gpu_mib") is not None}

    summary = _load_json(root / latest["checkpoint"] / "run-summary.json") if "train" in pipeline else None
    scores = _load_json(root / latest["results_dir"] / run["run"] / "summary.json") or []
    return {
        "run": run["run"],
        "track": latest["track"],
        "gpus": latest.get("gpus"),
        "stage_seconds": seconds,
        "preprocessing_seconds": preprocessing,
        "train_seconds": train,
        "total_seconds": None if preprocessing is None or train is None else preprocessing + train,
        "eval_seconds": seconds.get("eval"),
        "peak_gpu_mib": peaks,
        "supervised_tokens": (summary or {}).get("supervised_tokens"),
        "benchmarks": {
            s["benchmark"]: {key: s.get(key) for key in ("pass@1", "length", "truncation_rate", "n_generations")}
            for s in scores
        },
    }


def _minutes(seconds) -> str:
    return "-" if seconds is None else f"{seconds / 60:.1f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def render(rows: list[dict]) -> str:
    stage_names = [n for n in (*PREPROCESSING, "train", "eval") if any(n in r["stage_seconds"] for r in rows)]
    benchmarks = sorted({name for row in rows for name in row["benchmarks"]})
    cost = _table(
        ["run", "preprocessing (min)", "train (min)", "total (min)", "eval (min)", "supervised tokens",
         "peak GPU train (GiB)"],
        [[r["run"], _minutes(r["preprocessing_seconds"]), _minutes(r["train_seconds"]), _minutes(r["total_seconds"]),
          _minutes(r["eval_seconds"]), "-" if r["supervised_tokens"] is None else f"{r['supervised_tokens']:,}",
          f"{r['peak_gpu_mib']['train'] / 1024:.1f}" if "train" in r["peak_gpu_mib"] else "-"] for r in rows],
    )
    per_stage = _table(
        ["run", *(f"{n} (min)" for n in stage_names)],
        [[r["run"], *(_minutes(r["stage_seconds"].get(n)) for n in stage_names)] for r in rows],
    )
    generation = _table(
        ["run", *(f"{b}: tokens / truncated / pass@1" for b in benchmarks)],
        [[r["run"], *(
            f"{s['length']:.0f} / {s['truncation_rate']:.0%} / {s['pass@1']:.1%}" if (s := r["benchmarks"].get(b))
            else "-" for b in benchmarks)] for r in rows],
    )
    return (
        f"## Cost\n\n{cost}\n\n## Wall time per stage\n\n{per_stage}\n\n"
        f"## Generated tokens per answer (mean), truncation rate, accuracy\n\n{generation}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--efficiency-dir", default="rebuttal/efficiency")
    parser.add_argument("--root", default=".", help="directory the recorded checkpoint / results paths are under")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    directory = Path(args.efficiency_dir)
    runs = load_runs(directory)
    if not runs:
        raise SystemExit(f"no stage records in {directory}")
    shared = {r["latest"]["track"]: r["stages"]["prepare"] for r in runs if "prepare" in r["stages"]}
    rows = [summarize_run(run, shared, Path(args.root)) for run in runs]
    rows.sort(key=lambda r: (r["track"], r["run"]))
    (directory / "table.json").write_text(json.dumps(rows, indent=2))
    table = render(rows)
    (directory / "table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
