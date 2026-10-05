#!/usr/bin/env python3
"""Build data/raw/*.jsonl from local HF dataset dumps (AIME24/25, AMC12, MATH-500)."""
import json
import os
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("PALIGN_ASSET_ROOT", "/mnt/local/aiskylimit_new_nothing/P-ALIGN"))
RAW = Path(os.environ.get("PALIGN_DATA_DIR", str(ROOT / "data"))) / "raw"
DATASETS_ROOT = ASSET_ROOT / "datasets"

SPECS = [
    ("aime24", DATASETS_ROOT / "AIME_2024", "train", 30, "aime24.jsonl"),
    ("aime25", DATASETS_ROOT / "aime_2025", "train", 30, "aime25.jsonl"),
    ("amc12", DATASETS_ROOT / "aimo-validation-amc", "train", 83, "amc12.jsonl"),
    ("math500", DATASETS_ROOT / "MATH-500", "test", 500, "math500.jsonl"),
]


def _pick(row, keys):
    lower = {k.lower(): k for k in row}
    for want in keys:
        if want in row and row[want] not in (None, ""):
            return str(row[want])
        if want.lower() in lower and row[lower[want.lower()]] not in (None, ""):
            return str(row[lower[want.lower()]])
    return None


def _load_local(local_dir, split):
    candidates = [local_dir]
    if (local_dir / "data").is_dir():
        candidates.append(local_dir / "data")
    for path in candidates:
        for s in (split, "train", "test", "validation"):
            try:
                return load_dataset(str(path), split=s)
            except Exception:
                continue
    return None


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    qkeys = ["problem", "question", "input", "content", "Problem"]
    akeys = ["answer", "target", "solution", "ground_truth", "Answer"]
    for name, local_dir, split, expected, out_name in SPECS:
        out = RAW / out_name
        if out.exists():
            print(f"{name}: exists {out}")
            continue
        if not local_dir.exists():
            raise RuntimeError(f"missing local dataset {local_dir} and {out}")
        ds = _load_local(local_dir, split)
        if ds is None:
            raise RuntimeError(f"failed to load {local_dir}")
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
