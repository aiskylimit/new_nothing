#!/usr/bin/env python3
"""Pick the layer of a layer sweep: one run per layer (results_<prefix><layer>/eval/metrics.json), each judged on the
val sets of scripts/make_val_sets.py at one alpha. Score = mean DSR over val_jailbreak and val_refused (Llama Guard);
ties -> lower refusal rate on val_benign, then the lower layer. The val_benign refusal rate is reported, not used.

    python scripts/select_layer.py --results-root results --prefix sweep_selot_wf1_L --layers 10-22 --alpha 1.0 \
        --out results/sweep_selot_wf1

Writes <out>/sweep.md, <out>/sweep.json and <out>/best_layer.txt (the layer number); exits 1 if a run is missing.
When a run has results_<prefix><layer>/analysis.json (stage analyze), sweep.md also gets the activation analysis per
layer (reported, not used for the choice): mean ||alpha g(h)|| at the last prompt token per val set (absolute and
relative to ||h||) and the transport match of the fitting jailbreaks vs refused (Gaussian W2^2, covariance alignment,
before -> after steering).
"""

import argparse
import json
import sys
from pathlib import Path

DSR_SETS = ("val_jailbreak", "val_refused")


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def at(metrics, name, alpha, key):
    e = metrics["sets"].get(name, {}).get("per_alpha", {}).get(str(float(alpha)), {})
    return e.get(key)


def score_rows(results_root: Path, prefix: str, layers, alpha: float):
    rows, missing = [], []
    for l in layers:
        f = results_root / f"results_{prefix}{l}" / "eval" / "metrics.json"
        if not f.exists():
            missing.append(str(f))
            continue
        m = json.loads(f.read_text())
        dsr = {n: at(m, n, alpha, "dsr") for n in DSR_SETS}
        if any(v is None for v in dsr.values()):
            missing.append(f"{f} (no DSR at alpha {alpha})")
            continue
        rows.append({"layer": l, **{f"dsr_{n}": dsr[n] for n in DSR_SETS},
                     "dsr_mean": sum(dsr.values()) / len(dsr),
                     "refusal_val_benign": at(m, "val_benign", alpha, "refusal_rate"),
                     "base": {n: at(m, n, 0.0, "dsr") for n in DSR_SETS} | {"refusal_val_benign":
                                                                            at(m, "val_benign", 0.0, "refusal_rate")}})
    return rows, missing


def analysis_row(results_root: Path, prefix: str, layer: int, alpha: float, layer_key=None):
    """the analysis numbers of one sweep run at one alpha, or None (no analysis.json). The run is
    results_<prefix><layer>; layer_key: the steered layer when <layer> is something else (e.g. k of a k sweep)"""
    f = results_root / f"results_{prefix}{layer}" / "analysis.json"
    if not f.exists():
        return None
    a, key = json.loads(f.read_text()), str(float(alpha))
    layer = layer if layer_key is None else layer_key
    out = {}
    for name, d in a.get("displacement", {}).items():
        e = (d["per_layer"].get(str(layer)) or {}).get(key)
        if e:
            out[f"disp_{name}"] = e["last_token"]["mean"]
            out[f"rel_{name}"] = e["last_token"]["relative_mean"]
    t = (a.get("transport", {}).get(str(layer)) or {}).get(key)
    if t:
        for tag in ("before", "after"):
            out[f"w2_{tag}"], out[f"cov_{tag}"] = t[tag]["w2_sq"], t[tag]["cov_alignment"]
    return out


def best(rows):
    return min(rows, key=lambda r: (-round(r["dsr_mean"], 6), r["refusal_val_benign"] or 0.0, r["layer"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, missing = score_rows(Path(a.results_root), a.prefix, parse_layers(a.layers), a.alpha)
    if missing:
        sys.exit("missing sweep results:\n  " + "\n  ".join(missing))
    for r in rows:
        r["analysis"] = analysis_row(Path(a.results_root), a.prefix, r["layer"], a.alpha)
    b = best(rows)
    base = next((r["base"] for r in rows if r["base"]["val_jailbreak"] is not None), None)
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"
    md = [f"# Layer sweep ({a.prefix}*, alpha = {a.alpha:g}): best layer {b['layer']}", "",
          "Score = mean DSR over val_jailbreak and val_refused (Llama Guard 3, val split of the fit sets); "
          "val_benign refusal (refusal prefix) is reported, not used.", "",
          "| layer | DSR val_jailbreak | DSR val_refused | **mean DSR** | refusal val_benign |", "|---|---|---|---|---|"]
    if base:
        md.append(f"| alpha 0 (unsteered) | {pct(base['val_jailbreak'])} | {pct(base['val_refused'])} | "
                  f"{pct((base['val_jailbreak'] + base['val_refused']) / 2)} | {pct(base['refusal_val_benign'])} |")
    for r in rows:
        mark = " **(best)**" if r is b else ""
        md.append(f"| {r['layer']}{mark} | {pct(r['dsr_val_jailbreak'])} | {pct(r['dsr_val_refused'])} | "
                  f"**{pct(r['dsr_mean'])}** | {pct(r['refusal_val_benign'])} |")
    if any(r["analysis"] for r in rows):
        sets = ("val_benign", "val_refused", "val_jailbreak")
        g = lambda r, k, f="{:.3g}": "-" if (r["analysis"] or {}).get(k) is None else f.format(r["analysis"][k])
        md += ["", f"## Activation analysis per layer (alpha = {a.alpha:g}; reported, not used for the choice)", "",
               "||alpha g(h)||: mean displacement at the last prompt token on the val prompts (relative to ||h|| in "
               "brackets); benign should move little, jailbreak a lot. W2^2 / cov alignment: fitting jailbreaks vs "
               "refused in the transport subspace, before -> after steering (W2^2 down, alignment up = transport "
               "matches the refusal distribution).", "",
               "| layer | disp benign | disp refused | disp jailbreak | W2^2 before -> after | cov align before -> after |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            disp = [f"{g(r, 'disp_' + n)} ({g(r, 'rel_' + n, '{:.1%}')})" for n in sets]
            md.append(f"| {r['layer']} | " + " | ".join(disp) + f" | {g(r, 'w2_before')} -> {g(r, 'w2_after')} | "
                      f"{g(r, 'cov_before', '{:.3f}')} -> {g(r, 'cov_after', '{:.3f}')} |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.md").write_text("\n".join(md) + "\n")
    (out / "sweep.json").write_text(json.dumps({"alpha": a.alpha, "best_layer": b["layer"], "rows": rows}, indent=1))
    (out / "best_layer.txt").write_text(f"{b['layer']}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
