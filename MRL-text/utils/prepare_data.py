"""Offline preparation of downloaded raw training and evaluation files."""

import argparse
import os
from pathlib import Path
import tempfile

import pandas as pd

from utils.mteb_data import DEFAULT_EVAL_DATA_DIR, DEFAULT_RAW_DATA_DIR, offline_evaluation, prepare_tasks
from utils.mteb_tasks import DEFAULT_EVAL_TASKS


TRAIN_RAW_PATH = Path("training/quora-duplicates/pair/train-00000-of-00001.parquet")
DEFAULT_TRAIN_CSV = DEFAULT_RAW_DATA_DIR.parent / "quora-duplicates-pair.csv"


@offline_evaluation
def prepare_training(raw_data_dir=DEFAULT_RAW_DATA_DIR, output_csv=DEFAULT_TRAIN_CSV, *, force=False):
    destination = Path(output_csv).expanduser().resolve()
    if destination.is_file() and not force:
        print(f"Training CSV already prepared: {destination}")
        return destination
    source = Path(raw_data_dir).expanduser().resolve() / TRAIN_RAW_PATH
    if not source.is_file():
        raise FileNotFoundError(f"Raw training data missing: {source}. Run bash download.sh on an online "
                                "machine and copy data/raw to this server.")
    frame = pd.read_parquet(source)
    if not {"anchor", "positive"}.issubset(frame.columns) or len(frame) < 2:
        raise ValueError("Raw Quora data must contain annotated anchor/positive pairs.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".quora-", suffix=".csv", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame[["anchor", "positive"]].to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {len(frame)} training pairs: {destination}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--training_only", action="store_true")
    mode.add_argument("--eval_only", action="store_true")
    parser.add_argument("--raw_data_dir", default=str(DEFAULT_RAW_DATA_DIR))
    parser.add_argument("--training_csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--eval_data_dir", default=str(DEFAULT_EVAL_DATA_DIR))
    parser.add_argument("--eval_tasks", nargs="+", default=list(DEFAULT_EVAL_TASKS))
    parser.add_argument("--eval_splits", nargs="+", default=["test"])
    parser.add_argument("--eval_languages", nargs="+", default=["eng"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.eval_only:
        prepare_training(args.raw_data_dir, args.training_csv, force=args.force)
    if not args.training_only:
        prepare_tasks(args.eval_data_dir, raw_data_dir=args.raw_data_dir, names=args.eval_tasks,
                      splits=args.eval_splits, languages=args.eval_languages, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
