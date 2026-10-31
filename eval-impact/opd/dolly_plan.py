"""Add Dolly evaluation to old paired plans without changing training budgets."""
import copy
from pathlib import Path


def add_dolly_eval(plan, batch_size=16, seeds=(10, 20, 30, 40, 50)):
    plan = copy.deepcopy(plan)
    if plan.get("math_eval_enabled"):
        raise ValueError("Use --no-dolly-eval for the separate opt-in math pipeline")
    if batch_size < 1 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Positive evaluation batch size and distinct decoding seeds required")
    eval_jobs = {}
    for case in plan["cases"]:
        if not case.get("train_dir") or case["domain"] != "dolly-dskd":
            continue
        train = Path(case["train_dir"])
        folder = train / "eval_dolly"
        case.update(eval_dir=str(folder), eval_kind="dolly")
        command = [plan["arguments"]["python_bin"], "-m", "opd.dolly_eval",
                   "--model", plan["student"], "--adapter", str(train / "checkpoint-pending"),
                   "--training-dir", str(train), "--data", str(Path(plan["project_root"]) / "data/opd/dolly-dskd/test.jsonl"),
                   "--out", str(folder), "--batch-size", str(batch_size), "--seeds", *map(str, seeds)]
        eval_jobs[str(train)] = {"stage": "eval", "name": case["name"], "command": command,
            "artifact": str(folder / "evaluation_manifest.json"), "eval_dir": str(folder),
            "eval_kind": "dolly", "method": case["method"], "checkpoint_from": str(train / "latest.json")}
    jobs = []
    for job in plan["jobs"]:
        if job.get("eval_kind") == "dolly":
            continue
        jobs.append(job)
        if job["stage"] == "train" and job.get("train_dir") in eval_jobs:
            jobs.append(eval_jobs[job["train_dir"]])
    plan["jobs"] = jobs
    plan["dolly_eval_enabled"] = bool(eval_jobs)
    return plan
