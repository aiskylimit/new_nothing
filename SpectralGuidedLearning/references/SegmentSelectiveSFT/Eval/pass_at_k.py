"""
pass_at_k.py - Tinh pass@k (unbiased estimator, Codex) tu output cua math_eval.py.

Ly do can file nay: "acc" trong *_metrics.json la mean_score[0] cua evaluate.py,
tuc chi tinh MAU DAU TIEN cua moi cau (cot 0 cua score matrix). Voi n_sampling=32
no KHONG phai pass@1 trung binh tren 32 mau. File nay doc lai truong "score"
(list bool dai n) cua tung cau trong *.jsonl va tinh:

    pass@k = E_cau[ 1 - C(n-c, k) / C(n, k) ]        (c = so mau dung, n = so mau)

pass@1 theo cong thuc nay = trung binh accuracy tren dung n mau da sinh.
AVG = trung binh cong don gian qua cac task (macro-average, trong so bang nhau).

    python pass_at_k.py outputs_<tag>                 # k=1
    python pass_at_k.py outputs_<tag> --k 1 4 8 32
"""
import argparse
import glob
import json
import os
from math import comb


def pass_at_k(n, c, k):
    # Unbiased estimator (Chen et al. 2021, Codex).
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_scores(jsonl_path):
    scores = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line).get("score")
            if s is not None:
                scores.append([bool(x) for x in s])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Thu muc Eval/outputs_<tag>")
    ap.add_argument("--k", type=int, nargs="+", default=[1])
    ap.add_argument("--json", default="", help="Ghi ket qua ra file JSON (mac dinh <root>/pass_at_k.json)")
    args = ap.parse_args()

    # Layout: <root>/<task>/<tag>/<prefix>_s0_e-1.jsonl (+ *_metrics.json canh do)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.root, "*", "*", "**", "*.jsonl"), recursive=True)):
        if f.endswith("_metrics.json"):
            continue
        task = os.path.relpath(f, args.root).split(os.sep)[0]
        scores = load_scores(f)
        if not scores:
            continue
        n_min = min(len(s) for s in scores)
        row = {"task": task, "num_questions": len(scores), "n_sampling": n_min, "file": os.path.relpath(f, args.root)}
        for k in args.k:
            if k > n_min:
                row["pass@%d" % k] = None
                continue
            # Cat ve n_min de moi cau cung n (evaluate.py pad neu thieu, o day khong pad).
            vals = [pass_at_k(n_min, sum(s[:n_min]), k) for s in scores]
            row["pass@%d" % k] = round(100.0 * sum(vals) / len(vals), 2)
        rows.append(row)

    if not rows:
        print("Khong thay *.jsonl nao co truong 'score' trong %s" % args.root)
        return

    avg = {}
    for k in args.k:
        key = "pass@%d" % k
        vals = [r[key] for r in rows if r[key] is not None]
        avg[key] = round(sum(vals) / len(vals), 2) if vals else None

    hdr = "  %-12s %6s %4s " + " ".join(["%9s"] * len(args.k))
    print(hdr % (("task", "#q", "n") + tuple("pass@%d" % k for k in args.k)))
    for r in rows:
        print(hdr % ((r["task"], r["num_questions"], r["n_sampling"])
                     + tuple("-" if r["pass@%d" % k] is None else "%.2f" % r["pass@%d" % k] for k in args.k)))
    print(hdr % (("AVG", "", "") + tuple("-" if avg["pass@%d" % k] is None else "%.2f" % avg["pass@%d" % k] for k in args.k)))

    out = args.json or os.path.join(args.root, "pass_at_k.json")
    with open(out, "w") as fh:
        json.dump({"k": args.k, "average": avg, "tasks": rows}, fh, indent=2)
    print("\n  JSON: %s" % out)


if __name__ == "__main__":
    main()
