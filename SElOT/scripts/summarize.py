#!/usr/bin/env python3
"""Stage `summary`: fit checks + eval table + displacement table -> console and <results_dir>/summary.md."""

import json
from pathlib import Path

from common import setup


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    cfg, paths, log = setup(__doc__)
    res = paths["results"]
    lines = [f"# {cfg['run_id']}", "", f"model: `{cfg['model']['path']}`  method: `{cfg['method']['name']}`  "
             f"positions: `{cfg['steer'].get('positions', 'prompt')}`", ""]

    fit = load(res / "fit_report.json")
    if fit:
        lines += ["## Fit (training activations, alpha = 1)", "",
                  "| layer | ||g|| benign^2 | ||g|| refused^2 | ||g|| jailbreak^2 | ||delta*||^2 | mean gap left | Prop. 2 |",
                  "|---|---|---|---|---|---|---|"]
        for layer, r in fit["per_layer"].items():
            fmt = lambda k: f"{r[k]:.4g}" if k in r else "-"
            lines.append(f"| {layer} | {fmt('g_sq_benign')} | {fmt('g_sq_refused')} | {fmt('g_sq_jailbreak')} | "
                         f"{fmt('delta_sq_jailbreak')} | {fmt('field_mean_gap_left')} | {r.get('prop2_holds', '-')} |")
        lines += ["", f"numerical checks: {'FAILED: ' + '; '.join(fit['failures']) if fit['failures'] else 'all passed'}", ""]

    fs = load(res / "fit_sets.json")
    if fs:
        lines += [f"## Fit sets (behaviour: {fs['behaviour']}, topic cap {fs['topic_cap']}, val 1/{fs['val_folds']})", "",
                  "| set | pool | kept | after cap | train | val | train weight range |", "|---|---|---|---|---|---|---|"]
        for name, s in fs["sets"].items():
            wr = "-" if not s["weight_range"] else f"{s['weight_range'][0]:.2f}-{s['weight_range'][1]:.2f}"
            lines.append(f"| {name} | {s['pool']} | {s['kept']} | {s['final']} | {s['train']} | {s['val']} | {wr} |")
        lines.append("")

    metrics = load(res / "eval" / "metrics.json")
    if metrics:
        cols = [("asr", "ASR", True), ("dsr", "DSR", True)] + \
               [(f"asr_{x}", f"ASR {x}", True) for x in metrics.get("extra_judges", [])] + [("strongreject", "StrongREJECT", False), ("refusal_rate", "refusal", True),
                ("compliance_rate", "compliance", True), ("refusal_rate_prefix", "refusal (prefix)", True),
                ("accuracy", "accuracy", True),
                ("macro_accuracy", "macro acc", True),
                # AlpacaEval: only these three here; metrics.json keeps reward, reward_delta and the other win rates
                ("win_rate_vs_base", "win vs a0", True), ("lc_win_rate_vs_ref", "LC win vs ref", True),
                ("avg_words", "words", False)]
        sr = " + StrongREJECT" if metrics.get("strongreject") else ""
        orj = metrics.get("overrefusal_judge")
        sr += f"; over-refusal: XSTest 3-class judge {Path(orj).name}" if orj else "; over-refusal: refusal prefix"
        lines += [f"## Eval (ASR judge: {metrics['judge']}{sr})", "",
                  "| set | metric | alpha | n | " + " | ".join(c[1] for c in cols) + " |",
                  "|---|---|---|---|" + "---|" * len(cols)]
        for name, s in metrics["sets"].items():
            for alpha, e in s["per_alpha"].items():
                def fmt(k, pct):
                    v = e.get(k)
                    if v is None:
                        return "-"
                    return f"{100 * v:.1f}%" if pct else (f"{v:.1f}" if k == "avg_words" else f"{v:.3f}")
                lines.append(f"| {name} | {s.get('metric', s['kind'])} | {alpha} | {e['n']} | "
                             + " | ".join(fmt(k, p) for k, _, p in cols) + " |")
        lines.append("")

    analysis = load(res / "analysis.json")
    if analysis and analysis["displacement"]:
        a = str(analysis["alphas"][-1])
        lines += [f"## Displacement ||A h + b - h|| at alpha = {a} (held-out prompts)", "",
                  "| set | layer | last token mean | last token p95 | all prompt tokens mean | all p95 |", "|---|---|---|---|---|---|"]
        for name, d in analysis["displacement"].items():
            for layer, e in d["per_layer"].items():
                lt, at = e[a]["last_token"], e[a]["all_prompt_tokens"]
                lines.append(f"| {name} | {layer} | {lt['mean']:.4g} | {lt['p95']:.4g} | {at['mean']:.4g} | {at['p95']:.4g} |")
        lines.append("")
    if analysis and analysis["transport"]:
        lines += ["## Transport match (fitting jailbreaks vs refused, transport subspace)", "",
                  "| layer | alpha | W2^2 before | W2^2 after | cov align before | cov align after |", "|---|---|---|---|---|---|"]
        for layer, per in analysis["transport"].items():
            for alpha, t in per.items():
                if t:
                    lines.append(f"| {layer} | {alpha} | {t['before']['w2_sq']:.4g} | {t['after']['w2_sq']:.4g} | "
                                 f"{t['before']['cov_alignment']:.3f} | {t['after']['cov_alignment']:.3f} |")
    text = "\n".join(lines) + "\n"
    (res / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
