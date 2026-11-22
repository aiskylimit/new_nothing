#!/usr/bin/env python3
"""Side-by-side table of several full eval runs (results_<run>/eval/metrics.json): the main number of every set at
alpha 0 (unsteered) and at each run's steered alpha, plus mean DSR over the attack sets and mean compliance over the
over-refusal sets.

    python scripts/compare_runs.py --results-root results --runs selot_wf1_L14 alphasteer_wf1 --alpha 1.0 \
        --out results/compare_wf1.md
"""

import argparse
import json
from pathlib import Path

MAIN = {"asr": ("dsr", "DSR"), "refusal": ("compliance_rate", "compliance"), "mmlu": ("accuracy", "accuracy"),
        "gsm8k": ("accuracy", "accuracy"), "alpaca_eval": ("win_rate_vs_base", "win rate vs alpha 0")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ms = {r: json.loads((Path(a.results_root) / f"results_{r}" / "eval" / "metrics.json").read_text()) for r in a.runs}
    first = ms[a.runs[0]]
    names = list(first["sets"])
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"

    def get(m, n, alpha, key):
        return m["sets"].get(n, {}).get("per_alpha", {}).get(str(float(alpha)), {}).get(key)

    md = [f"# Comparison at alpha = {a.alpha:g} (alpha 0 = unsteered model)", "",
          "| set | metric | alpha 0 | " + " | ".join(a.runs) + " |", "|---|---|---|" + "---|" * len(a.runs)]
    means = {"DSR (mean over attack sets)": [], "compliance (mean over safe over-refusal sets)": []}
    for n in names:
        s = first["sets"][n]
        key, label = MAIN.get(s["metric"], (None, None))
        if s["metric"] == "refusal" and s["kind"] == "harmful":
            key, label = "refusal_rate", "refusal (should stay high)"
        if not key:
            continue
        md.append(f"| {n} | {label} | {pct(get(first, n, 0.0, key))} | "
                  + " | ".join(pct(get(ms[r], n, a.alpha, key)) for r in a.runs) + " |")
        if s["metric"] == "asr" and n != "advbench_test":
            means["DSR (mean over attack sets)"].append(n)
        if s["metric"] == "refusal" and s["kind"] != "harmful":
            means["compliance (mean over safe over-refusal sets)"].append(n)
    for label, sets in means.items():
        key = "dsr" if label.startswith("DSR") else "compliance_rate"

        def mean(m, alpha):
            v = [get(m, n, alpha, key) for n in sets]
            v = [x for x in v if x is not None]
            return sum(v) / len(v) if v else None
        md.append(f"| **{label}** | | {pct(mean(first, 0.0))} | " + " | ".join(pct(mean(ms[r], a.alpha)) for r in a.runs)
                  + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
