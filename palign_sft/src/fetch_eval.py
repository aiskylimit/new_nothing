#!/usr/bin/env python3
"""Build normalized eval JSONL files from local Hugging Face snapshots."""

import json
import os
from pathlib import Path

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(
    os.environ.get(
        "PALIGN_SFT_ASSET_ROOT", "/mnt/local/aiskylimit_new_nothing/palign_sft"
    )
)
RAW = Path(os.environ.get("PALIGN_SFT_DATA_DIR", str(ROOT / "data"))) / "raw"
DATASETS_ROOT = ASSET_ROOT / "datasets"

SPECS = [
    ("aime24", DATASETS_ROOT / "AIME_2024", "train", 30, "aime24.jsonl"),
    ("aime25", DATASETS_ROOT / "aime_2025", "train", 30, "aime25.jsonl"),
    ("amc12", DATASETS_ROOT / "aimo-validation-amc", "train", 83, "amc12.jsonl"),
    ("math500", DATASETS_ROOT / "MATH-500", "test", 500, "math500.jsonl"),
]


def _validate_existing(path: Path, expected: int):
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid existing evaluation file {path}: {error}"
        ) from error
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not isinstance(
            row.get("answer"), str
        ):
            raise TypeError(f"{path}: row {index} lacks string question/answer fields")
    return len(rows)


def _pick(row, keys):
    lower = {key.lower(): key for key in row}
    for wanted in keys:
        actual = wanted if wanted in row else lower.get(wanted.lower())
        if actual is not None and row[actual] not in (None, ""):
            return str(row[actual])
    return None


def _load_local(local_dir: Path, split: str):
    data_root = local_dir / "data"
    search_root = data_root if data_root.is_dir() else local_dir
    parquet_files = sorted(search_root.rglob("*.parquet"))
    if parquet_files:
        from pyarrow import parquet

        rows = []
        for filename in parquet_files:
            rows.extend(parquet.read_table(filename).to_pylist())
        return rows

    if (local_dir / "state.json").exists() or (
        local_dir / "dataset_dict.json"
    ).exists():
        try:
            loaded = load_from_disk(str(local_dir))
            if hasattr(loaded, "keys"):
                for name in (split, "train", "test", "validation"):
                    if name in loaded:
                        return loaded[name]
            return loaded
        except Exception:  # noqa: BLE001, S110 - try the next local layout
            pass

    candidates = [local_dir]
    if (local_dir / "data").is_dir():
        candidates.append(local_dir / "data")
    for path in candidates:
        for name in (split, "train", "test", "validation"):
            try:
                return load_dataset(str(path), split=name)
            except Exception:  # noqa: BLE001, S112 - probe alternate split/layout
                continue

    for suffix, loader in (
        ("*.parquet", "parquet"),
        ("*.jsonl", "json"),
        ("*.json", "json"),
    ):
        files = sorted(str(path) for path in local_dir.rglob(suffix))
        if not files:
            continue
        try:
            return load_dataset(loader, data_files={split: files}, split=split)
        except Exception:  # noqa: BLE001, S112 - probe alternate file format
            continue
    return None


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    question_keys = ["problem", "question", "input", "content", "Problem"]
    answer_keys = ["answer", "target", "solution", "ground_truth", "Answer"]
    for name, local_dir, split, expected, output_name in SPECS:
        output = RAW / output_name
        if output.exists():
            count = _validate_existing(output, expected)
            print(f"{name}: validated {count} existing rows -> {output}")
            continue
        if not local_dir.exists():
            raise RuntimeError(f"missing local dataset {local_dir} and {output}")
        dataset = _load_local(local_dir, split)
        if dataset is None:
            raise RuntimeError(f"failed to load {local_dir}")
        rows = []
        for raw in dataset:
            row = dict(raw)
            question = _pick(row, question_keys)
            answer = _pick(row, answer_keys)
            if question is not None and answer is not None:
                rows.append({"question": question, "answer": answer})
        if len(rows) != expected:
            raise RuntimeError(
                f"{name}: expected {expected} valid rows, found {len(rows)}"
            )
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
        print(f"{name}: wrote {len(rows)} -> {output}")


if __name__ == "__main__":
    main()
