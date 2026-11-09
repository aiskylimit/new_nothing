"""Local snapshots only: never resolve a model/dataset name through the Hub."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def asset_root() -> Path:
    return Path(os.environ.get("CORED_ASSET_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()


def asset_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else asset_root() / path).resolve()


def local_model_path(value: str) -> str:
    if not value:
        raise ValueError("Pass --model_name with a local model directory.")
    path = asset_path(value)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Local model missing config.json: {path}. Provision models using cored/download.txt; "
            "pass models/<model-name> or an absolute path. Hub IDs are not supported."
        )
    return str(path)


def local_adapter_path(value: str) -> str:
    # Training outputs follow the caller's working directory, not the asset store.
    path = Path(value).expanduser().resolve()
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Local adapter missing adapter_config.json: {path}")
    return str(path)


def load_local_rows(value: str | Path, *, split: str | None = None) -> list[dict]:
    """Read raw snapshot parquet/JSON(L), or a datasets.save_to_disk directory.

    A split is selected from filenames (train.parquet, test-00000-..., etc.).
    Never concatenate train/test/validation or silently substitute a split.
    """
    path = asset_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset missing: {path}. Provision cored/download.txt first.")
    if path.is_dir() and ((path / "state.json").exists() or (path / "dataset_dict.json").exists()):
        from datasets import load_from_disk

        dataset = load_from_disk(str(path))
        if hasattr(dataset, "keys"):
            if split is None:
                raise ValueError(f"Choose a split for {path}")
            dataset = dataset[split]
        return list(dataset)
    if path.is_file():
        files = [path]
    else:
        # HF's default subset is often data/, beside alternative subsets such
        # as AIME2025 part1/part2. Match the baseline loader's default-subset rule.
        search_root = path / "data" if (path / "data").is_dir() else path
        candidates = sorted(p for p in search_root.rglob("*")
                            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
                            and p.suffix in {".parquet", ".jsonl", ".json"})
        if split is not None:
            pattern = re.compile(rf"^{re.escape(split)}(?:[-_.]|$)", re.IGNORECASE)
            candidates = [p for p in candidates if pattern.match(p.name)]
        # A snapshot may include dataset_info.json and alternative format exports.
        files = []
        for suffix in (".parquet", ".jsonl", ".json"):
            files = [p for p in candidates if p.suffix == suffix
                     and p.name not in {"dataset_info.json", "dataset_dict.json", "state.json"}]
            if files:
                break
    if not files:
        raise FileNotFoundError(f"No local {split or 'data'} parquet/JSON(L) files in {path}")
    rows = []
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows.extend(pq.read_table(file).to_pylist())
        elif file.suffix == ".jsonl":
            rows.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
        elif file.suffix == ".json":
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and split is not None:
                data = data[split]
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list of examples in {file}")
            rows.extend(data)
        else:
            raise ValueError(f"Unsupported local dataset format: {file}")
    if not rows:
        raise ValueError(f"Empty local dataset: {path}")
    return rows
