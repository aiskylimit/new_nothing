#!/usr/bin/env python3
"""Print and save the GRIT run summary: training curve + eval table + paired deltas.

Reads (all optional, whatever exists):
  <results-dir>/train/metrics.jsonl, <results-dir>/train/diagnostics_summary.json
  <results-dir>/eval/eval_results.json
Writes <results-dir>/summary.md and <results-dir>/summary.json and prints them.
"""

import argparse
import json
from pathlib import Path

TRAIN_COLUMNS = [
    ("step", "step", "{:d}"),
    ("reward_mean", "reward", "{:.3f}"),
    ("unsafe_fraction", "unsafe", "{:.3f}"),
    ("zero_advantage_group_fraction", "zero_adv", "{:.2f}"),
    ("response_tokens_mean", "resp_tok", "{:.0f}"),
    ("validation_unsafe_fraction", "val_unsafe", "{:.3f}"),
    ("validation_tokens_mean", "val_tok", "{:.0f}"),
    ("trust_kl_mean", "kl_mean", "{:.2e}"),
    ("trust_violation_fraction", "kl_viol", "{:.3f}"),
    ("correction_to_task_ratio", "corr/task", "{:.1e}"),
    ("relative_update_norm", "rel_upd", "{:.1e}"),
    ("peak_vram_gb", "vram_gb", "{:.0f}"),
    ("step_seconds", "step_s", "{:.0f}"),
]
# Harmful prompts: unsafe answers are the failure. Benign prompts: refusals are the failure (over-refusal).
HARMFUL = ["pku_safety", "advbench", "harmbench", "jailbreakbench", "sorrybench", "harmfulqa", "alert"]
BENIGN = ["xstest_safe", "orbench_hard", "alpacaeval"]
EVAL_COLUMNS = ["accuracy_%", "extracted_%", "unsafe_%", "refusal_%", "refusal_regex_%", "judge_parse_fail_%",
                "mean_tokens", "truncated_%", "prompt_too_long"]


def fmt(value, pattern="{:.2f}"):
    if value is None:
        return "-"
    try:
        return pattern.format(value)
    except (ValueError, TypeError):
        return str(value)


def table(headers, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, rows)])


def train_section(train_dir):
    metrics_path = train_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return None, {}
    steps = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    for row in steps:
        if row.get("validation_count"):
            row["validation_tokens_mean"] = row["validation_response_tokens"] / row["validation_count"]
    rows = [[fmt(row.get(key), pattern) for key, _, pattern in TRAIN_COLUMNS] for row in steps]
    text = "## Training (per step)\n\n" + table([name for _, name, _ in TRAIN_COLUMNS], rows)
    alerts = [(row["step"], row["alerts"]) for row in steps if row.get("alerts")]
    if alerts:
        text += "\n\nAlerts:\n" + "\n".join(f"- step {step}: {', '.join(names)}" for step, names in alerts)
    diagnostics_path = train_dir / "diagnostics_summary.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {}
    if diagnostics:
        text += "\n\nRun diagnostics:\n\n```json\n" + json.dumps(diagnostics, indent=2, sort_keys=True) + "\n```"
    return text, {"steps": len(steps), "last": steps[-1] if steps else None, "diagnostics": diagnostics}


def eval_section(eval_path):
    if not eval_path.is_file():
        return None, {}
    payload = json.loads(eval_path.read_text())
    present = [c for c in EVAL_COLUMNS if any(c in m for m in payload["metrics"])]
    rows = [[m["model"], m["benchmark"], m["n"], *[fmt(m.get(c)) for c in present]] for m in payload["metrics"]]
    text = "## Evaluation\n\n" + table(["model", "benchmark", "n", *present], rows)
    by_key = {(m["model"], m["benchmark"]): m for m in payload["metrics"]}
    harmful = [b for b in HARMFUL if any(key[1] == b for key in by_key)]
    benign = [b for b in BENIGN if any(key[1] == b for key in by_key)]
    if harmful or benign:
        headers = ["model", *[f"unsafe_% {b}" for b in harmful], *[f"over_refusal_% {b}" for b in benign]]
        rows = [[model, *[fmt(by_key.get((model, b), {}).get("unsafe_%")) for b in harmful],
                 *[fmt(by_key.get((model, b), {}).get("refusal_%")) for b in benign]]
                for model in payload["models"]]
        text += ("\n\n## Safety vs over-refusal (lower is better everywhere)\n\n" + table(headers, rows)
                 + "\n\nunsafe_% on harmful prompts; over_refusal_% = WildGuard response refusal on benign "
                   "prompts that look harmful (XSTest safe, OR-Bench hard-1k) or are plain instructions (AlpacaEval).")
    if payload["paired_deltas"]:
        delta_rows = [[d["model"], d["benchmark"], d["metric"], fmt(d["delta_pp"]),
                       f"[{fmt(d['ci95_pp'][0])}, {fmt(d['ci95_pp'][1])}]", d["base_only"], d["model_only"]]
                      for d in payload["paired_deltas"]]
        text += (f"\n\n## Model - {payload['reference']} (percentage points, paired bootstrap 95% CI)\n\n"
                 + table(["model", "benchmark", "metric", "delta_pp", "ci95_pp",
                          f"{payload['reference']}_only", "model_only"], delta_rows))
    text += ("\n\nunsafe_% = allenai/wildguard \"Harmful response: yes\"; refusal_% = \"Response refusal: yes\"; "
             "judge_parse_fail_% = WildGuard output without a yes/no for either (counted as neither).")
    return text, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--title", default="GRIT run summary")
    args = parser.parse_args()
    root = Path(args.results_dir)
    train_text, train = train_section(root / "train")
    eval_text, evaluation = eval_section(root / "eval" / "eval_results.json")
    parts = [f"# {args.title}", train_text or "_No training metrics found._", eval_text or "_No eval results found._"]
    markdown = "\n\n".join(parts) + "\n"
    (root / "summary.md").write_text(markdown)
    (root / "summary.json").write_text(json.dumps({"train": train, "eval": evaluation}, indent=2, ensure_ascii=False))
    print(markdown, flush=True)
    print(f"Wrote {root / 'summary.md'} and {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
