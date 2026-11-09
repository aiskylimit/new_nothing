#!/usr/bin/env python3
"""Convert a local s1K-1.1 snapshot into the two baseline SFT datasets."""

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
TARGETS = {
    "label": "solution",
    "longcot": "deepseek_thinking_trajectory",
}


def _read_json(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("train", "data", "rows"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"unsupported JSON structure in {path}")


def _dataset_files(source: Path, suffix: str):
    data_root = source / "data"
    root = data_root if data_root.is_dir() else source
    return [str(path) for path in sorted(root.rglob(f"*{suffix}"))]


def load_rows(source: Path) -> Iterable[Mapping]:
    """Load a file, `save_to_disk` directory, or downloaded HF snapshot."""
    if not source.exists():
        raise FileNotFoundError(f"s1K-1.1 source does not exist: {source}")
    if source.is_file():
        if source.suffix not in {".json", ".jsonl"}:
            raise ValueError(f"unsupported source file: {source}")
        return _read_json(source)

    # Hub snapshots of s1K-1.1 contain ordinary Parquet under data/. Reading it
    # directly avoids network resolution and a writable Hugging Face cache.
    parquet_files = _dataset_files(source, ".parquet")
    if parquet_files:
        from pyarrow import parquet

        rows = []
        for filename in parquet_files:
            rows.extend(parquet.read_table(filename).to_pylist())
        return rows

    for suffix in (".jsonl", ".json"):
        json_files = _dataset_files(source, suffix)
        if json_files:
            rows = []
            for filename in json_files:
                rows.extend(_read_json(Path(filename)))
            return rows

    from datasets import load_dataset, load_from_disk

    errors = []
    if (source / "state.json").exists() or (source / "dataset_dict.json").exists():
        try:
            loaded = load_from_disk(str(source))
            return (
                loaded["train"]
                if hasattr(loaded, "keys") and "train" in loaded
                else loaded
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            errors.append(f"load_from_disk: {exc}")

    try:
        return load_dataset(str(source), split="train")
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        errors.append(f"directory loader: {exc}")

    for suffix, loader in ((".arrow", "arrow"),):
        files = _dataset_files(source, suffix)
        if not files:
            continue
        try:
            return load_dataset(loader, data_files={"train": files}, split="train")
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            errors.append(f"{loader} loader: {exc}")

    details = "\n  - ".join(errors) if errors else "no supported data files found"
    raise RuntimeError(
        f"could not load local s1K-1.1 snapshot at {source}:\n  - {details}"
    )


def build_records(rows, target_field: str):
    records = []
    invalid = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        question = row.get("question")
        target = row.get(target_field)
        if not isinstance(question, str) or not question.strip():
            invalid.append((index, "question"))
            continue
        if not isinstance(target, str) or not target.strip():
            invalid.append((index, target_field))
            continue
        records.append(
            {
                "instruction": INSTRUCTION,
                "input": question.strip(),
                "output": target.strip(),
            }
        )
    if invalid:
        preview = ", ".join(f"row {i}: {field}" for i, field in invalid[:10])
        extra = f" (+{len(invalid) - 10} more)" if len(invalid) > 10 else ""
        raise ValueError(f"missing or empty required fields: {preview}{extra}")
    return records


def write_jsonl(records, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--expected-rows", type=int, default=1000)
    args = parser.parse_args()

    rows = [dict(row) for row in load_rows(args.source)]
    if args.expected_rows > 0 and len(rows) != args.expected_rows:
        raise SystemExit(
            f"expected {args.expected_rows} source rows, found {len(rows)} in {args.source}"
        )

    outputs = {
        "label": args.output_dir / "s1k_label.jsonl",
        "longcot": args.output_dir / "s1k_longcot.jsonl",
    }
    for name, target_field in TARGETS.items():
        records = build_records(rows, target_field)
        if args.expected_rows > 0 and len(records) != args.expected_rows:
            raise SystemExit(
                f"{name}: expected {args.expected_rows} valid rows, found {len(records)}"
            )
        write_jsonl(records, outputs[name])
        mean_chars = sum(len(row["output"]) for row in records) / max(len(records), 1)
        print(
            f"{name}: wrote {len(records)} rows from `{target_field}` "
            f"(mean target chars={mean_chars:.1f}) -> {outputs[name]}"
        )


if __name__ == "__main__":
    main()
