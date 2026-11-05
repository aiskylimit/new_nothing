"""Attach additional instruction evaluations to existing training cases."""
import copy
from pathlib import Path
from .benchmarks import ALL_BENCHMARKS


def add_instruction_eval(plan, benchmarks=ALL_BENCHMARKS, batch_size=16, seeds=(10, 20, 30, 40, 50)):
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or set(benchmarks) - set(ALL_BENCHMARKS):
        raise ValueError("Select distinct supported instruction benchmarks")
    if batch_size < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive batch size and distinct decoding seeds required")
    if plan.get("math_eval_enabled"):
        raise ValueError("Instruction benchmarks require the Dolly pipeline")
    benchmarks = tuple(dict.fromkeys(("dolly", *benchmarks)))
    plan = copy.deepcopy(plan)
    plan["instruction_benchmarks"] = list(benchmarks)
    plan["benchmark_cases"] = []
    extra = {}
    for case in plan["cases"]:
        if case["domain"] != "dolly-dskd" or not case.get("train_dir"):
            continue
        train = Path(case["train_dir"])
        for name in benchmarks:
            if name == "dolly":
                continue
            folder = train / "eval_benchmarks" / name
            name_job = f"eval-{name}-{case['domain']}-{case['method']}-{case['seed']}"
            plan["benchmark_cases"].append({**case, "name": name_job, "benchmark": name,
                                            "eval_dir": str(folder), "eval_kind": "instruction"})
            command = [plan["arguments"]["python_bin"], "-m", "opd.dolly_eval",
                "--benchmark", name, "--model", plan["student"],
                "--adapter", str(train / "checkpoint-pending"), "--training-dir", str(train),
                "--data", str(Path(plan["project_root"]) / "data/opd/benchmarks" / name / "test.jsonl"),
                "--out", str(folder), "--batch-size", str(batch_size), "--seeds", *map(str, seeds)]
            extra.setdefault(str(train), []).append({"stage": "eval", "name": name_job,
                "command": command, "artifact": str(folder / "evaluation_manifest.json"),
                "eval_dir": str(folder), "eval_kind": "instruction", "benchmark": name,
                "method": case["method"], "checkpoint_from": str(train / "latest.json")})
    jobs = []
    for job in plan["jobs"]:
        if job.get("eval_kind") == "instruction":
            continue
        jobs.append(job)
        if job["stage"] == "train":
            jobs.extend(extra.get(job.get("train_dir"), []))
    plan["jobs"] = jobs
    return plan
