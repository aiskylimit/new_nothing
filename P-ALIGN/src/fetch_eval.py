#!/usr/bin/env python3
"""Download public eval splits into data/raw/*.jsonl (AIME24/25, AMC12, MATH-500)."""
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SPECS = [
    ("aime24", "Maxwell-Jia/AIME_2024", "train", 30, "aime24.jsonl"),
    ("aime25", "yentinglin/aime_2025", "train", 30, "aime25.jsonl"),
    ("amc12", "AI-MO/aimo-validation-amc", "train", 83, "amc12.jsonl"),
    ("math500", "HuggingFaceH4/MATH-500", "test", 500, "math500.jsonl"),
]


def _pick(row, keys):
    lower = {k.lower(): k for k in row}
    for want in keys:
        if want in row and row[want] not in (None, ""):
            return str(row[want])
        if want.lower() in lower and row[lower[want.lower()]] not in (None, ""):
            return str(row[lower[want.lower()]])
    return None


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    qkeys = ["problem", "question", "input", "content", "Problem"]
    akeys = ["answer", "target", "solution", "ground_truth", "Answer"]
    for name, hf_id, split, expected, out_name in SPECS:
        ds = None
        for s in (split, "train", "test", "validation"):
            try:
                ds = load_dataset(hf_id, split=s)
                break
            except Exception:
                continue
        if ds is None:
            raise RuntimeError(f"failed to load {hf_id}")
        out = RAW / out_name
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for row in ds:
                q, a = _pick(dict(row), qkeys), _pick(dict(row), akeys)
                if q is None or a is None:
                    continue
                f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                n += 1
        print(f"{name}: wrote {n} (expected {expected}) -> {out}")


if __name__ == "__main__":
    main()
