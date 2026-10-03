#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset


TRAIN_COLUMNS = ("problem", "solution", "Question", "Answer")
EVAL_DATASETS = ("aime25", "aime26", "hmmt25")


def parquet_files(path: Path) -> list[str]:
    # Hugging Face repositories may include additional dataset configurations
    # beside the default data directory, such as AIME25 part1 and part2.
    search_root = path / "data" if (path / "data").is_dir() else path
    files = sorted(str(file) for file in search_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return files


def load_parquet_dataset(path: Path) -> Dataset:
    tables = [pq.read_table(file) for file in parquet_files(path)]
    table = pa.concat_tables(tables).replace_schema_metadata(None)
    return Dataset(table)


def save_dataset(dataset: Dataset, destination: Path, overwrite: bool):
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}. Pass --overwrite to replace it.")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))
    print(f"Saved {len(dataset):,} rows to {destination}")


def prepare_training_dataset(raw_root: Path, output_root: Path, overwrite: bool):
    source = raw_root / "train"
    dataset = load_parquet_dataset(source)
    missing = [column for column in TRAIN_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    dataset = dataset.select_columns(list(TRAIN_COLUMNS))
    dataset = dataset.filter(
        lambda row: all(isinstance(row[column], str) and row[column].strip() for column in TRAIN_COLUMNS),
        desc="Removing incomplete training rows",
    )
    save_dataset(dataset, output_root / "train", overwrite)


def prepare_eval_dataset(name: str, raw_root: Path, output_root: Path, overwrite: bool):
    source = raw_root / "eval" / name
    dataset = load_parquet_dataset(source)
    if "problem" not in dataset.column_names or "answer" not in dataset.column_names:
        raise ValueError(f"{name} must contain problem and answer columns: {dataset.column_names}")

    id_column = "problem_idx" if "problem_idx" in dataset.column_names else "id"

    def normalize(row, index):
        return {
            "problem_id": row.get(id_column, index),
            "problem": row["problem"],
            "answer": str(row["answer"]),
        }

    dataset = dataset.map(normalize, with_indices=True, remove_columns=dataset.column_names)
    save_dataset(dataset, output_root / "eval" / name, overwrite)


def main():
    parser = argparse.ArgumentParser(description="Prepare all OPSD datasets for offline use.")
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    prepare_training_dataset(raw_root, output_root, args.overwrite)
    for name in EVAL_DATASETS:
        prepare_eval_dataset(name, raw_root, output_root, args.overwrite)


if __name__ == "__main__":
    main()
