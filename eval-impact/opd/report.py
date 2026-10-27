"""Aggregate native evaluation outputs across explicitly labeled runs/seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .data import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=EVAL_DIR")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    groups, rows = {}, []
    for spec in args.run:
        label, folder = spec.split("=", 1)
        paths = sorted(Path(folder).rglob("results*.json"))
        if len(paths) != 1:
            raise ValueError(f"Specify one eval directory per seed, got {len(paths)} result files: {folder}")
        obj = json.loads(paths[0].read_text())
        manifest = json.loads((Path(folder) / "evaluation_manifest.json").read_text())
        for task in ("minerva_math500", "gsm8k_cot"):
            resolved_task = manifest.get("task_map", {}).get(task, task)
            metrics = obj["results"][resolved_task]
            wanted = "exact_match,flexible-extract" if task == "gsm8k_cot" else "exact_match,none"
            if wanted not in metrics:
                raise ValueError(f"Missing native {wanted} for {task}: {paths[0]}")
            row = {"label": label, "task": task, "seed": manifest["seed"],
                   "metric": wanted, "score": metrics[wanted], "file": str(paths[0]),
                   "scope": manifest["scope"], "eval_protocol": {k: manifest[k] for k in
                        ("max_new_tokens", "num_fewshot", "lm_eval_version", "limit", "dtype")}}
            rows.append(row)
            groups.setdefault((label, task), []).append(row)
    summary = []
    for (label, task), values in groups.items():
        if len({r["seed"] for r in values}) != len(values):
            raise ValueError(f"Repeated seed for {label}/{task}")
        if len({json.dumps(r["eval_protocol"], sort_keys=True) for r in values}) != 1:
            raise ValueError(f"Inconsistent evaluation protocols for {label}/{task}")
        x = np.array([v["score"] for v in values], dtype=float)
        std = float(x.std(ddof=1)) if len(x) > 1 else None
        half = float(t.ppf(0.975, len(x) - 1) * std / np.sqrt(len(x))) if std is not None else None
        summary.append({"label": label, "task": task, "n_seeds": len(x), "mean": float(x.mean()),
                        "std": std, "ci95_half_width": half, "scope": values[0]["scope"]})
    save_json(args.out, {"runs": rows, "summary": summary})
    lines = ["# OPD evaluation", "", "Run | Task | Seeds | Mean | Std | 95% CI half-width | Scope",
             "--- | --- | --- | --- | --- | --- | ---"]
    for row in summary:
        lines.append(" | ".join(str(row[k]) for k in
                     ("label", "task", "n_seeds", "mean", "std", "ci95_half_width", "scope")))
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
