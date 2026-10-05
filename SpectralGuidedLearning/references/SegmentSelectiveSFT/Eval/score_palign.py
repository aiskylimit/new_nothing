"""
score_palign.py - Cham lai output cua math_eval.py DUNG NHU P-ALIGN (src/evaluation.py + src/report.py).

Ly do: grader.py (kieu Qwen-math) va math_verify cua P-ALIGN co the cham lech nhau vai cau
(142 vs 142.0, bieu thuc LaTeX...). De so sanh truc tiep voi bang cua P-ALIGN, file nay:
  - lay ca n mau sinh ra cua moi cau (truong "code" trong *.jsonl cua math_eval.py),
  - dap an goc = truong "answer" tho cua test.jsonl (P-ALIGN: str(item["answer"]).strip()),
    thieu thi dung "gt",
  - cham bang math_verify: verify(parse("$" + gold + "$"), parse(pred)), timeout 10s cho ca cau
    (het gio -> moi mau cua cau do tinh sai), giong het label_with_math_verify cua P-ALIGN,
  - pass@1 = trung binh (so mau dung / so mau) moi cau; pass@3 (pass@n) = 1 neu co mau dung;
    AVG = trung binh cong 4 bo, trong so bang nhau (report.py).

    python score_palign.py outputs_<tag>
Ghi <root>/palign_scored/<task>.jsonl (label, passn, output_ans tung cau) va <root>/palign_score.json.
Moi task co nhieu file output (vd --quick va chay day du) thi lay file moi nhat.
"""
import argparse
import glob
import json
import os
import signal

from math_verify import parse, verify
from tqdm import tqdm


# --- chep nguyen tu P-ALIGN src/evaluation.py ---
def timeout(seconds=10):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Verification timed out.")

        def wrapper(*args, **kwargs):
            if os.name != "posix":
                return func(*args, **kwargs)
            old = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

        return wrapper

    return decorator


@timeout(seconds=10)
def label_with_math_verify(preds, golden):
    parsed_preds = list(map(parse, preds))
    parsed_golden = list(map(parse, ["$" + golden + "$"] * len(preds)))
    try:
        labels = list(map(verify, parsed_golden, parsed_preds))
    except Exception:
        labels = [0] * len(preds)
    return [int(x) for x in labels], parsed_preds
# --- het phan chep ---


def score_file(path, out_path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    results = []
    for item in tqdm(rows, desc=os.path.basename(path)):
        outputs = item.get("code")
        if not outputs:
            continue
        if isinstance(outputs, str):
            outputs = [outputs]
        answer = str(item.get("answer", item.get("gt", ""))).strip()
        try:
            labels, parsed = label_with_math_verify(outputs, answer)
        except Exception:
            labels, parsed = [0] * len(outputs), [""] * len(outputs)
        results.append({"idx": item.get("idx"), "answer": answer, "label": labels,
                        "passn": int(any(labels)), "output_ans": parsed})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return results


def metrics(results):
    # Giong report.py cua P-ALIGN.
    n = len(results)
    if n == 0:
        return 0.0, 0.0
    p1 = sum(sum(r["label"]) / len(r["label"]) for r in results if r["label"])
    pn = sum(1 for r in results if r["passn"])
    return 100.0 * p1 / n, 100.0 * pn / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Thu muc Eval/outputs_<tag>")
    args = ap.parse_args()

    # Layout nhu pass_at_k.py: <root>/<task>/<tag>/<prefix>.jsonl (+ *_metrics.json).
    latest = {}
    for f in glob.glob(os.path.join(args.root, "*", "*", "**", "*.jsonl"), recursive=True):
        task = os.path.relpath(f, args.root).split(os.sep)[0]
        if task not in latest or os.path.getmtime(f) > os.path.getmtime(latest[task]):
            latest[task] = f
    if not latest:
        raise SystemExit("Khong thay output *.jsonl nao trong %s" % args.root)

    tasks = {}
    for task in sorted(latest):
        f = latest[task]
        results = score_file(f, os.path.join(args.root, "palign_scored", task + ".jsonl"))
        p1, pn = metrics(results)
        n = min((len(r["label"]) for r in results), default=0)
        tasks[task] = {"num_questions": len(results), "n_sampling": n, "pass@1": p1,
                       "pass@%d" % n: pn, "file": os.path.relpath(f, args.root)}

    n_all = sorted({t["n_sampling"] for t in tasks.values()})
    keys = ["pass@1"] + ["pass@%d" % n for n in n_all if n > 1]
    avg = {}
    for k in keys:
        vals = [t[k] for t in tasks.values() if k in t]
        avg[k] = round(sum(vals) / len(vals), 2) if vals else None
    # AVG tinh tu so chua lam tron (nhu report.py), roi moi lam tron tung task.
    for t in tasks.values():
        for k in keys:
            if k in t:
                t[k] = round(t[k], 2)

    hdr = "  %-10s %5s %3s " + " ".join(["%8s"] * len(keys))
    print(hdr % (("task", "#q", "n") + tuple(keys)))
    for task, t in tasks.items():
        print(hdr % ((task, t["num_questions"], t["n_sampling"])
                     + tuple("%.2f" % t[k] if k in t else "-" for k in keys)))
    print(hdr % (("AVG", "", "") + tuple("-" if avg[k] is None else "%.2f" % avg[k] for k in keys)))

    out = os.path.join(args.root, "palign_score.json")
    with open(out, "w") as fh:
        json.dump({"scorer": "math_verify (P-ALIGN)", "k": [int(k[5:]) for k in keys],
                   "average": avg, "tasks": tasks}, fh, indent=2)
    print("\n  JSON: %s" % out)


if __name__ == "__main__":
    main()
