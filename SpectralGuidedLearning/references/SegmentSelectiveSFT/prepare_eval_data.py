"""
prepare_eval_data.py - Chuyen bo benchmark da tai ve (dang HF dataset, xem
downloads.txt) thanh data/<task>/test.jsonl ma Eval/math_eval.py doc duoc.

    python prepare_eval_data.py --data-root /mnt/local/_data/<project>

Mapping (ten thu muc tai ve -> ten task cua Eval/):
    aime24                -> data/aime24/test.jsonl   (math-ai/aime24, split test)
    aime25                -> data/aime25/test.jsonl   (math-ai/aime25, split test)
    MATH-500              -> data/math500/test.jsonl  (HuggingFaceH4/MATH-500, split test)
    aimo-validation-amc   -> data/amc12/test.jsonl    (AI-MO/aimo-validation-amc, split train,
                                                        83 cau AMC12 2022-2023)

Moi dong ghi: idx, question, answer (+ solution neu co). Eval/parser.py lay
question tu 'question', dap an tu 'answer' (aime24/aime25/amc12) hoac tu
\\boxed{} cuoi cua 'solution' (math500) - ca hai deu co o day.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))

# (thu muc tai ve, ten task, split, cot cau hoi, cot dap an)
BENCHES = [
    ("aime24", "aime24", "test", "problem", "answer"),
    ("aime25", "aime25", "test", "problem", "answer"),
    ("MATH-500", "math500", "test", "problem", "answer"),
    ("aimo-validation-amc", "amc12", "train", "problem", "answer"),
]


def last_boxed(text):
    # Lay noi dung \boxed{...} cuoi cung, dem ngoac de khong cat giua chung.
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def norm_answer(a):
    a = str(a).strip()
    # aimo-validation-amc ghi dap an dang so thuc "227.0" -> "227"
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def load_rows(path, split):
    from datasets import load_dataset, load_from_disk
    try:
        return load_dataset(path, split=split)
    except Exception as e1:
        try:
            ds = load_from_disk(path)
            return ds[split] if hasattr(ds, "keys") and split in ds else ds
        except Exception as e2:
            sys.exit(f"Khong doc duoc {path}: load_dataset -> {e1} | load_from_disk -> {e2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Thu muc chua aime24/ aime25/ MATH-500/ aimo-validation-amc/")
    ap.add_argument("--out-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--only", nargs="*", default=[], help="Chi lam vai task (ten task: aime24 amc12 ...)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for src_name, task, split, qk, ak in BENCHES:
        if args.only and task not in args.only:
            continue
        src = os.path.join(args.data_root, src_name)
        out = os.path.join(args.out_root, task, "test.jsonl")
        if os.path.exists(out) and not args.overwrite:
            print(f"  {task:8s} da co {out} - bo qua (--overwrite de lam lai)")
            continue
        if not os.path.isdir(src):
            print(f"  {task:8s} KHONG thay {src} - bo qua", file=sys.stderr)
            continue
        rows = load_rows(src, split)
        cols = set(rows.column_names)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        n_boxed = 0
        with open(out, "w") as fh:
            for i, r in enumerate(rows):
                q = r[qk] if qk in cols else r["question"]
                ans = r.get(ak) if ak in cols else None
                sol = r.get("solution") if "solution" in cols else None
                if ans is None or str(ans).strip() == "":
                    # math-ai/aime24 co the chi co dap an trong \boxed{} cua solution
                    ans = last_boxed(sol or "")
                    n_boxed += 1
                if ans is None:
                    sys.exit(f"{task} cau {i}: khong co dap an ('{ak}' rong, solution khong co \\boxed)")
                rec = {"idx": i, "question": q, "answer": norm_answer(ans)}
                if sol:
                    rec["solution"] = sol
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        note = f" ({n_boxed} cau lay dap an tu \\boxed)" if n_boxed else ""
        print(f"  {task:8s} {len(rows):4d} cau  <- {src_name} [{split}] -> {out}{note}")


if __name__ == "__main__":
    main()
