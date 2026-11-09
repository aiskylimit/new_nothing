"""Ablation cắt CoT — sinh dữ liệu train với các mức cắt KHÁC adaptive k* (không cần vLLM/GPU).

Đầu vào: data/train_s1k.jsonl (CoT đầy đủ, có ``steps``). Cách dựng bản ghi GIỐNG HỆT
proposal_v2/prefix_truncate.py (steps[:k], thinking = "\\n\\n".join(steps[:k]), các trường
orig_n_steps / prefix_n_steps / prefix_frac / prefix_sufficient) để mọi thứ phía sau
(keystep_select, phase1, compute_weights, phase2_distill) chạy y hệt.

Các biến thể (out_dir/<tên>.jsonl):
    full   frac = 1.00 cho mọi mẫu (không cắt)
    f025   k = ceil(0.25 * n_steps)
    f050   k = ceil(0.50 * n_steps)
    f075   k = ceil(0.75 * n_steps)
    rand   cắt NGẪU NHIÊN khớp phân phối với k*: lấy tập tỉ lệ k*/n_steps của adaptive
           (data/train_s1k_prefixed.jsonl), hoán vị ngẫu nhiên (seed) cho các mẫu -> phân
           phối tỉ lệ cắt (trung bình, phân vị) y hệt adaptive nhưng KHÔNG phụ thuộc mẫu.

    python -m proposal_v2.prefix_variants --out_dir data/ablation_trunc
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random


def _write(path, rows, ks):
    fr = []
    with open(path, "w") as w:
        for r, k in zip(rows, ks):
            ns = len(r["steps"])
            k = max(1, min(ns, k))
            new_steps = r["steps"][:k]
            out = dict(r)
            out["steps"] = new_steps
            out["thinking"] = "\n\n".join(new_steps)
            out["orig_n_steps"] = ns
            out["prefix_n_steps"] = k
            out["prefix_frac"] = round(k / max(ns, 1), 3)
            out["prefix_sufficient"] = k < ns
            fr.append(k / max(ns, 1))
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
    fr.sort()
    n = len(fr)
    print(f"  {os.path.basename(path):12s} n={n} frac mean={sum(fr)/n:.3f} "
          f"p25={fr[n//4]:.3f} p50={fr[n//2]:.3f} p75={fr[3*n//4]:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/train_s1k.jsonl")
    ap.add_argument("--adaptive", default="data/train_s1k_prefixed.jsonl")
    ap.add_argument("--out_dir", default="data/ablation_trunc")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.inp)]
    ns = [len(r["steps"]) for r in rows]
    print(f"[variants] {len(rows)} mẫu, n_steps TB={sum(ns)/len(ns):.1f}", flush=True)

    _write(os.path.join(args.out_dir, "full.jsonl"), rows, ns)
    for name, frac in (("f025", 0.25), ("f050", 0.50), ("f075", 0.75)):
        _write(os.path.join(args.out_dir, f"{name}.jsonl"), rows, [math.ceil(frac * n) for n in ns])

    ada = [json.loads(l) for l in open(args.adaptive)]
    ratios = [a["prefix_n_steps"] / max(a["orig_n_steps"], 1) for a in ada]
    rng = random.Random(args.seed)
    pool = ratios[:]
    rng.shuffle(pool)
    while len(pool) < len(rows):             # mẫu bị loại khỏi adaptive (vd s1k_8): bốc thêm ngẫu nhiên
        pool.append(rng.choice(ratios))
    ks = [round(pool[i] * ns[i]) for i in range(len(rows))]
    _write(os.path.join(args.out_dir, "rand.jsonl"), rows, ks)
    _write(os.path.join(args.out_dir, "_adaptive_ref.jsonl"), ada, [a["prefix_n_steps"] for a in ada])


if __name__ == "__main__":
    main()
