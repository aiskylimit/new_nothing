"""Report instruction scores by benchmark and average only complete suites."""
from .benchmarks import BENCHMARKS
from .dolly_results import read_dolly_result, dolly_rows, paired_dolly_deltas


def collect_benchmark_results(plan, dolly):
    selected = plan.get("instruction_benchmarks", ["dolly"])
    rows = [{**r, "benchmark": "dolly"} for r in dolly]
    seeds, missing = [], []
    for case in plan.get("benchmark_cases", []):
        result = read_dolly_result(case)
        if result is None:
            missing.append(case["name"])
            continue
        if result.get("kind") != case["benchmark"] or result["protocol"].get("benchmark") != case["benchmark"]:
            raise ValueError("Benchmark result belongs to a different dataset")
        row, per_seed = dolly_rows(case, result)
        rows.append({**row, "benchmark": case["benchmark"]})
        seeds.extend({**r, "benchmark": case["benchmark"]} for r in per_seed)
    deltas = []
    for name in selected:
        deltas.extend({**r, "benchmark": name} for r in paired_dolly_deltas([r for r in rows if r["benchmark"] == name]))
    groups = {}
    for row in rows:
        key = (row["domain"], row["training_seed"], row["method"])
        if row["benchmark"] in groups.setdefault(key, {}):
            raise ValueError("Duplicate benchmark result")
        groups[key][row["benchmark"]] = row
    wide = []
    for (domain, seed, method), observations in groups.items():
        complete = all(name in observations for name in selected)
        present = list(observations.values())
        # All scores in an average must describe one checkpoint and one decode protocol.
        comparable = len({(r["checkpoint_step"], r["training_steps_budget"]) for r in present}) == 1
        for key in ("decoding_seeds", "batch_size", "max_prompt_length", "max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "dtype"):
            comparable = comparable and all(r["protocol"].get(key) == present[0]["protocol"].get(key) for r in present)
        full = complete and comparable and all(r["training_complete"] and r["scope"].startswith("full_") for r in present)
        row = {"domain": domain, "training_seed": seed, "method": method,
               "checkpoint_step": present[0]["checkpoint_step"], "complete_suite": full}
        for name in selected:
            row[name] = observations[name]["rougeL_mean"] if name in observations else None
        row["avg_rougeL"] = sum(row[name] for name in selected) / len(selected) if full else None
        wide.append(row)
    return {"benchmarks": selected, "metrics": rows, "per_seed": seeds,
            "paired_deltas": deltas, "wide": wide, "missing": missing,
            "average_note": "Unweighted mean of selected benchmark ROUGE-L scores; requires full tests, completed training, same checkpoint and decoding protocol. UnNI is excluded."}


def markdown_table(suite):
    names = suite["benchmarks"]
    labels = {"dolly": "Dolly", **{k: v["label"] for k, v in BENCHMARKS.items()}}
    lines = ["", "Instruction benchmarks — ROUGE-L (0–100). UnNI excluded.",
             " | ".join(["Method", "Training seed", *[labels[n] for n in names], "Avg."]),
             " | ".join(["---"] * (len(names) + 3))]
    for row in suite["wide"]:
        values = [f"{row[n]:.3f}" if row[n] is not None else "pending" for n in [*names, "avg_rougeL"]]
        lines.append(" | ".join([row["method"], str(row["training_seed"]), *values]))
    lines += [suite["average_note"]]
    return lines
