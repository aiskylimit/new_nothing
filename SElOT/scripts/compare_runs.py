#!/usr/bin/env python3
"""Final result of full eval runs (results_<run>/eval/metrics.json), as tables only, one row per method (the unsteered
model = alpha 0 of the first run, then each run at its steered alpha):

  Safety       DSR (%) per attack set by Llama Guard 3 (main judge), the same by the HarmBench classifier, and the
               StrongREJECT score (lower = safer); Avg = mean over the sets shown
  Utility      over-refusal: compliance (%) on XSTest safe / OR-Bench hard / OKTest (XSTest 3-class judge), refusal (%)
               on XSTest unsafe (should stay high); tasks: AlpacaEval win rate vs the unsteered model and
               length-controlled win rate vs the reference outputs (Skywork-Reward-V2), MMLU and GSM8K accuracy

--log-out writes the tuning log next to it: the layer sweep (sweep.md: val DSR, over-refusal and activation analysis
per layer) and every run's summary.md (fit checks, fit sets, every metric at every alpha, displacement, transport
match), plus where the responses and the logs are.

    python scripts/compare_runs.py --results-root results --runs selot_wf1_L14 alphasteer_wf1 --alpha 1.0 \
        --out results/compare_wf1.md [--log-out results/tune_log_wf1.md --sweep-md results/sweep_selot_wf1/sweep.md \
        --work <WORK>]
"""

import argparse
import json
import re
from pathlib import Path

SAFETY = [("advbench_test", "AdvBench"), ("attack_aim", "AIM"), ("attack_autodan", "AutoDAN"),
          ("attack_cipher", "Cipher"), ("attack_gcg", "GCG"), ("attack_jailbroken", "Jailbroken"),
          ("attack_pair", "PAIR"), ("attack_renellm", "ReNeLLM")]
OVERREFUSAL = [("xstest_safe", "XSTest safe", "compliance_rate", "compliance ↑"),
               ("orbench_hard", "OR-Bench hard", "compliance_rate", "compliance ↑"),
               ("oktest", "OKTest", "compliance_rate", "compliance ↑"),
               ("xstest_unsafe", "XSTest unsafe", "refusal_rate", "refusal ↑")]
TASKS = [("alpaca_eval", "AlpacaEval", "win_rate_vs_base", "win vs unsteered"),
         ("alpaca_eval", "AlpacaEval", "lc_win_rate_vs_ref", "LC win vs ref"),
         ("mmlu", "MMLU", "accuracy", "acc ↑"), ("gsm8k", "GSM8K", "accuracy", "acc ↑")]


def method_label(run):
    m = re.match(r"selot(_gated(?:_(logistic|single))?)?_wf\d.*?_L(\d+)((?:_[a-z_]+?\d+)*)$", run)
    if m:
        tuned = re.findall(r"_([a-z_]+?)(\d+)", m.group(4))
        name = {None: "SElOT", "logistic": "SElOT-gated (logistic gate, one OT)",
                "single": "SElOT-gated (GMM gate, one OT)"}.get(m.group(2), "SElOT")
        name = "SElOT-gated (GMM gate, mixture OT)" if m.group(1) == "_gated" else name
        return f"{name}, layer {m.group(3)}" + "".join(f", {p} = {v}" for p, v in tuned) if m.group(1) else \
            f"{name} (layer {m.group(3)}" + "".join(f", {p} = {v}" for p, v in tuned) + ")"
    if run.startswith("alphasteer"):
        return "AlphaSteer"
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-out", default=None)
    ap.add_argument("--sweep-md", default=None)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    root = Path(a.results_root)
    ms = {r: json.loads((root / f"results_{r}" / "eval" / "metrics.json").read_text()) for r in a.runs}
    rows = [("Unsteered (α = 0)", ms[a.runs[0]], 0.0)] + [(method_label(r), ms[r], a.alpha) for r in a.runs]

    def get(m, name, alpha, key):
        return m["sets"].get(name, {}).get("per_alpha", {}).get(str(float(alpha)), {}).get(key)

    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"
    num = lambda v: "-" if v is None else f"{v:.3f}"

    def table(title, cols, value, fmt, avg=True):
        """cols: [(set, header)]; value(m, set, alpha) -> float or None"""
        head = "| Method | " + " | ".join(h for _, h in cols) + (" | **Avg** |" if avg else " |")
        out = [f"### {title}", "", head, "|---|" + "---|" * (len(cols) + bool(avg))]
        for label, m, alpha in rows:
            vals = [value(m, n, alpha) for n, _ in cols]
            cells = [fmt(v) for v in vals]
            if avg:
                got = [v for v in vals if v is not None]
                cells.append("**" + fmt(sum(got) / len(got) if got else None) + "**")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out + [""]

    present = lambda cols: [c for c in cols if any(c[0] in m["sets"] for m in ms.values())]
    safety = present(SAFETY)

    def dsr_hb(m, n, alpha):
        v = get(m, n, alpha, "asr_harmbench")
        return None if v is None else 1.0 - v

    md = [f"# Results (steered: alpha = {a.alpha:g}; unsteered: alpha = 0)", "",
          "## Safety", ""]
    md += table("DSR (%) ↑, Llama Guard 3", safety, lambda m, n, al: get(m, n, al, "dsr"), pct)
    if any("harmbench" in m.get("extra_judges", []) for m in ms.values()):
        md += table("DSR (%) ↑, HarmBench classifier", safety, dsr_hb, pct)
    if any(m.get("strongreject") for m in ms.values()):
        md += table("StrongREJECT score ↓", safety, lambda m, n, al: get(m, n, al, "strongreject"), num)
    md += ["## Utility", ""]
    orf = [c for c in OVERREFUSAL if any(c[0] in m["sets"] for m in ms.values())]
    md += table("Over-refusal (%)", [(n, f"{h} ({k})") for n, h, _, k in orf],
                lambda m, n, al: get(m, n, al, dict((x[0], x[2]) for x in orf)[n]), pct, avg=False)
    tasks = [c for c in TASKS if any(c[0] in m["sets"] for m in ms.values())]
    md += table("Tasks (%)", [(f"{n}|{key}", f"{h} {k}") for n, h, key, k in tasks],
                lambda m, nk, al: get(m, nk.split("|")[0], al, nk.split("|")[1]), pct, avg=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md))
    print("\n".join(md))

    if a.log_out:
        log = [f"# Tuning log: {', '.join(a.runs)}", "",
               f"Final tables: `{a.out}`. Below: the layer sweep, then each run's summary (fit checks, fit sets, every "
               "metric at every alpha, activation displacement, transport match).", ""]
        if a.sweep_md and Path(a.sweep_md).exists():
            log += ["---", "", Path(a.sweep_md).read_text(), ""]
        for r in a.runs:
            s = root / f"results_{r}" / "summary.md"
            log += ["---", "", s.read_text() if s.exists() else f"# {r}\n\n(no summary.md)", ""]
            log += [f"Files of `{r}`: metrics `{root / f'results_{r}' / 'eval' / 'metrics.json'}`, logs "
                    f"`{root / f'results_{r}' / 'logs'}`"
                    + (f", responses and every judge's labels `{Path(a.work) / 'generations' / r}/*.judged.jsonl`"
                       if a.work else ""), ""]
        Path(a.log_out).write_text("\n".join(log))
        print(f"tuning log: {a.log_out}")


if __name__ == "__main__":
    main()
