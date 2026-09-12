"""Add distractor schema units to generator training rows.

Only ``generation_train.jsonl`` is rewritten. Every other file of the grounding
directory (selector data, dev/test generator data, schemas, manifests) is copied
unchanged, so the output directory can replace the input directory in the
existing preparation pipeline via ``CYPHER_GROUNDING_INPUT_DIR``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cypher_extract.paths import get_data_root  # noqa: E402
from schema_grounding.augmentation import AugmentationConfig, augment_generation_row  # noqa: E402

TRAIN_FILE = "generation_train.jsonl"
AUGMENTATION_MANIFEST = "augmentation_manifest.json"


def parse_args() -> argparse.Namespace:
    defaults = AugmentationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = get_data_root()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "cypherbench_schema_grounding_full_final",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "cypherbench_schema_grounding_distractor_v1",
    )
    parser.add_argument("--gold-ratio", type=float, default=defaults.gold_ratio)
    parser.add_argument("--full-ratio", type=float, default=defaults.full_ratio)
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=defaults.max_distractors,
        help="Upper bound on added units per noisy row, counting relationship-endpoint closure.",
    )
    parser.add_argument("--neighbor-weight", type=float, default=defaults.neighbor_weight)
    parser.add_argument("--lexical-weight", type=float, default=defaults.lexical_weight)
    parser.add_argument("--random-weight", type=float, default=defaults.random_weight)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    # Match the grounding pipeline's serialization (sorted keys) byte for byte.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_output_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input-dir and output-dir must differ")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "mean": round(statistics.mean(ordered), 3),
        "max": ordered[-1],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(row["schema_augmentation"]["mode"] for row in rows)
    noisy_added = [
        len(row["schema_augmentation"]["distractor_unit_ids"])
        for row in rows
        if row["schema_augmentation"]["mode"] == "noisy"
    ]
    schema_chars = [len(json.dumps(row["sub_schema"], ensure_ascii=False, indent=2)) for row in rows]
    return {
        "rows": len(rows),
        "modes": dict(sorted(modes.items())),
        "noisy_added_units": _distribution(noisy_added),
        "noisy_added_unit_histogram": dict(sorted(Counter(noisy_added).items())),
        "sub_schema_chars": _distribution(schema_chars),
    }


def main() -> None:
    args = parse_args()
    config = AugmentationConfig(
        gold_ratio=args.gold_ratio,
        full_ratio=args.full_ratio,
        max_distractors=args.max_distractors,
        neighbor_weight=args.neighbor_weight,
        lexical_weight=args.lexical_weight,
        random_weight=args.random_weight,
        seed=args.seed,
    )
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    for name in (TRAIN_FILE, "schemas.jsonl"):
        if not (input_dir / name).is_file():
            raise FileNotFoundError(input_dir / name)

    schemas = {row["schema_id"]: row for row in read_jsonl(input_dir / "schemas.jsonl")}
    rows = [
        augment_generation_row(row, schemas[row["schema_id"]], config)
        for row in read_jsonl(input_dir / TRAIN_FILE)
    ]

    prepare_output_dir(input_dir, output_dir, args.overwrite)
    for path in input_dir.iterdir():
        if path.is_file() and path.name != TRAIN_FILE:
            shutil.copy2(path, output_dir / path.name)
    write_jsonl(output_dir / TRAIN_FILE, rows)

    manifest = {
        "input_directory": str(input_dir),
        "config": asdict(config),
        "rewritten_files": [TRAIN_FILE],
        "train": summarize(rows),
    }
    (output_dir / AUGMENTATION_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
