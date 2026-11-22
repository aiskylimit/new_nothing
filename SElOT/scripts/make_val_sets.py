#!/usr/bin/env python3
"""Eval-format files from the val split of this model's fit sets, for picking the layer (and alpha) without touching
the test sets. Writes <out>/val_{jailbreak,refused,benign}.jsonl and <out>/data.yaml (data.fit of --data-config +
these three eval sets, also listed as analysis.sets for the activation analysis):

    val_jailbreak  D_j val    metric asr      DSR (Llama Guard): the jailbreaks the unsteered model answers
    val_refused    D_r val    metric asr      DSR: the vanilla harmful prompts it refuses (steering must keep that)
    val_benign     D_b val    metric refusal  refusal rate (refusal prefix): over-refusal on benign prompts

    python scripts/make_val_sets.py --fitsets <work>/fitsets/<tag>/wf1/llamaguard --out <work>/valsets/... \
        [--data-config configs/data/v1.yaml] [--benign-max 1000] [--max-new-tokens 128]
"""

import argparse
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fitsets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-config", default=str(ROOT / "configs/data/v1.yaml"))
    ap.add_argument("--benign-max", type=int, default=1000, help="fixed random subset of D_b val (seed 0)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = ap.parse_args()
    fs, out = Path(a.fitsets), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = []
    for name, metric, kind in (("jailbreak", "asr", "harmful"), ("refused", "asr", "harmful"),
                               ("benign", "refusal", "benign")):
        rows = [json.loads(l) for l in open(fs / f"{name}.jsonl") if l.strip()]
        rows = [r for r in rows if r.get("split") == "val"]
        if not rows:
            raise SystemExit(f"no val rows in {fs / name}.jsonl: run the label stage first")
        if name == "benign" and len(rows) > a.benign_max:
            keep = set(random.Random(0).sample(range(len(rows)), a.benign_max))
            rows = [r for i, r in enumerate(rows) if i in keep]
        path = out / f"val_{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                rec = {"id": r["id"], "prompt": r["prompt"], "topic": r.get("topic"), "source": r.get("source")}
                if metric == "asr":
                    rec["behavior"] = r.get("redteam_query") or r["prompt"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        specs.append({"name": f"val_{name}", "path": str(path), "metric": metric, "kind": kind,
                      "max_new_tokens": a.max_new_tokens})
        print(f"val_{name}: {len(rows)} prompts -> {path}")
    fit = (yaml.safe_load(open(a.data_config)).get("data") or {}).get("fit")
    # the same held-out prompts for the activation analysis (stage analyze: displacement per group, transport match)
    analysis = [{"name": s["name"], "path": s["path"], "group": s["name"][4:]} for s in specs]
    (out / "data.yaml").write_text(yaml.safe_dump({"data": {"fit": fit, "eval": specs},
                                                   "analysis": {"sets": analysis}}, sort_keys=False))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
