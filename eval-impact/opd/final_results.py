"""Combine accuracy, paired improvements, runtime, and BI into a final report."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from .data import save_json, read_jsonl

TASKS = {"minerva_math500": "exact_match,none", "gsm8k_cot": "exact_match,flexible-extract"}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def evaluation(case):
    folder = Path(case["eval_dir"])
    files = list(folder.rglob("results*.json"))
    manifest_path = folder / "evaluation_manifest.json"
    if not files or not manifest_path.is_file():
        return None
    if len(files) != 1:
        raise ValueError(f"Expected one result file per evaluation: {folder}")
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(files[0].read_text())
    protocol = {key: manifest.get(key) for key in
                ("max_new_tokens", "num_fewshot", "lm_eval_version", "limit", "dtype", "decoding")}
    rows = []
    for task, metric in TASKS.items():
        local_task = manifest.get("task_map", {}).get(task, task)
        score = result["results"][local_task][metric]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid accuracy in {files[0]}")
        effective = result.get("n-samples", {}).get(local_task, {}).get("effective")
        rows.append({"domain": case["domain"], "method": case["method"], "seed": case["seed"],
                     "task": task, "accuracy_percent": score * 100,
                     "scope": manifest["scope"], "n_eval_samples": effective,
                     "result_file": str(files[0]), "protocol": protocol})
    return rows


def training_cost(case):
    if not case.get("train_dir"):
        return None
    root = Path(case["train_dir"])
    metrics = read_jsonl(root / "metrics.jsonl") if (root / "metrics.jsonl").is_file() else []
    if not metrics:
        return None
    last = metrics[-1]
    # Current trainer records cumulative fields across resumes.
    elapsed = last.get("cumulative_loop_seconds")
    tokens = last.get("cumulative_generated_tokens")
    if elapsed is None or tokens is None:
        # Do not silently add session counters incorrectly for older checkpoints.
        elapsed, tokens = None, None
    peak = [m["peak_memory_gb"] for m in metrics if m.get("peak_memory_gb") is not None]
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    return {"domain": case["domain"], "method": case["method"], "seed": case["seed"],
            "optimizer_steps": last["step"], "loop_seconds": elapsed, "generated_tokens": tokens,
            "peak_allocated_gb": max(peak) if peak else None,
            "last_opd_loss": last.get("opd_loss"), "last_impact_loss": last.get("impact_loss"),
            "state": status.get("state", "unknown")}


def aggregate(plan_file, out):
    plan = json.loads(Path(plan_file).read_text())
    out = Path(out)
    rows, missing, costs = [], [], []
    for case in plan["cases"]:
        observed = evaluation(case)
        if observed is None:
            missing.append(case["name"])
        else:
            rows.extend(observed)
        cost = training_cost(case)
        if cost:
            costs.append(cost)
    lookup = {(r["domain"], r["seed"], r["task"], r["method"]): r for r in rows}
    deltas = []
    for (domain, seed, task, method), row in lookup.items():
        if method != "opd_impact":
            continue
        base = lookup.get((domain, seed, task, "opd"))
        if not base:
            continue
        if base["protocol"] != row["protocol"] or base["scope"] != row["scope"]:
            raise ValueError(f"Paired evaluations use different protocols: {domain}/{seed}/{task}")
        deltas.append({"domain": domain, "seed": seed, "task": task,
                       "opd_percent": base["accuracy_percent"], "opd_impact_percent": row["accuracy_percent"],
                       "delta_percentage_points": row["accuracy_percent"] - base["accuracy_percent"],
                       "scope": row["scope"]})
    groups = {}
    for row in rows:
        groups.setdefault((row["domain"], row["method"], row["task"], row["scope"]), []).append(row)
    summary = []
    for (domain, method, task, scope), observations in groups.items():
        if len({r["seed"] for r in observations}) != len(observations):
            raise ValueError("Duplicate evaluation seeds in aggregation")
        if len({json.dumps(r["protocol"], sort_keys=True) for r in observations}) > 1:
            raise ValueError("Cannot combine different eval protocols across seeds")
        values = [r["accuracy_percent"] for r in observations]
        summary.append({"domain": domain, "method": method, "task": task, "n_seeds": len(values),
                        "mean_accuracy_percent": statistics.mean(values),
                        "std_accuracy_percent": statistics.stdev(values) if len(values) > 1 else None,
                        "scope": scope})
    bi_layers, bi_pairs = [], []
    for path in sorted((Path(plan["run_root"]) / "bi").glob("*/summary.json")):
        obj = json.loads(path.read_text())
        for layer in obj["layers"]:
            bi_layers.append({"calibration": path.parent.name, **layer})
    pair_file = Path(plan["run_root"]) / "bi/comparisons.json"
    if pair_file.exists():
        for label, metrics in json.loads(pair_file.read_text()).items():
            bi_pairs.append({"comparison": label, **{k: v for k, v in metrics.items() if k != "per_layer"}})
    missing_artifacts = [j["name"] for j in plan.get("jobs", []) if not Path(j["artifact"]).is_file()]
    budget_complete = all(c["optimizer_steps"] >= plan.get("arguments", {}).get("steps", 0) for c in costs)
    payload = {"schema_version": 1, "comparison": plan["comparison"], "profile": plan["profile"],
               "state": "complete" if not missing and not missing_artifacts and budget_complete else "incomplete",
               "missing_evaluations": missing, "missing_artifacts": missing_artifacts,
               "training_budget_complete": budget_complete,
               "accuracy": rows, "summary": summary, "paired_deltas": deltas,
               "training_cost": costs, "bi_comparisons": bi_pairs,
               "cost_note": "Loop seconds and generated tokens include resumed training sessions; total job wall time is in jobs.json."}
    save_json(out / "final_results.json", payload)
    write_csv(out / "accuracy_by_run.csv", [{k: v for k, v in r.items() if k != "protocol"} for r in rows])
    write_csv(out / "accuracy_summary.csv", summary)
    write_csv(out / "opd_vs_impact_delta.csv", deltas)
    write_csv(out / "training_cost.csv", costs)
    write_csv(out / "bi_all_layers.csv", bi_layers)
    write_csv(out / "bi_comparisons.csv", bi_pairs)
    lines = ["# OPD vs OPD + IMPACT — final results", "", f"Status: {payload['state']}",
             f"Profile: {plan['profile']}", "", "Domain | Method | Task | Seeds | Accuracy (%) | Std | Scope",
             "--- | --- | --- | --- | --- | --- | ---"]
    for row in summary:
        std = f"{row['std_accuracy_percent']:.3f}" if row['std_accuracy_percent'] is not None else "n/a"
        lines.append(f"{row['domain']} | {row['method']} | {row['task']} | {row['n_seeds']} | {row['mean_accuracy_percent']:.3f} | {std} | {row['scope']}")
    lines += ["", "Domain | Seed | Task | OPD (%) | OPD+IMPACT (%) | Delta (pp)", "--- | --- | --- | --- | --- | ---"]
    for row in deltas:
        lines.append(f"{row['domain']} | {row['seed']} | {row['task']} | {row['opd_percent']:.3f} | {row['opd_impact_percent']:.3f} | {row['delta_percentage_points']:+.3f}")
    lines += ["", f"BI calibration sets: {len({r['calibration'] for r in bi_layers})}; paired BI comparisons: {len(bi_pairs)}.",
              "Detailed costs, all BI layer ranks/weights and per-run accuracy are in the adjacent CSV/JSON files.",
              "One training seed has no across-seed variability estimate. Missing results are never filled with zero."]
    if missing:
        lines += ["", "Missing evaluations: " + ", ".join(missing)]
    if missing_artifacts:
        lines += ["", "Missing artifacts: " + ", ".join(missing_artifacts)]
    (out / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(out / "FINAL_RESULTS.md", flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    aggregate(args.plan, args.out)


if __name__ == "__main__":
    main()
