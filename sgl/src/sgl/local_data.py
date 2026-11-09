"""Read provisioned dataset snapshots without Hub access or cache writes."""

import json
from pathlib import Path


def load_source_dataset(name, *, split="train", revision=None):
    source = Path(name)
    if not source.exists():
        from datasets import load_dataset

        return load_dataset(name, split=split, revision=revision)
    if (source / "state.json").exists() or (source / "dataset_dict.json").exists():
        from datasets import load_from_disk

        loaded = load_from_disk(str(source))
        return loaded[split] if hasattr(loaded, "keys") else loaded
    root = source / "data" if (source / "data").is_dir() else source
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else []
    if files:
        from pyarrow import parquet

        return [row for file in files for row in parquet.read_table(file).to_pylist()]
    files = [source] if source.is_file() else sorted(root.rglob("*.jsonl"))
    if files:
        return [json.loads(line) for file in files
                for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"No supported local dataset files in {source}")
