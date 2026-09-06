"""Prepare prompt/response JSONL files for joint generator and selector training.

Train and eval files are pre-interleaved so batches contain both tasks whenever
the available row counts allow it. Rows are never duplicated. Do not shuffle
individual rows again in the training data loader, or the batch-level task mix
will no longer be guaranteed. Test files remain separated by task.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distillation.data_cache import preparation_fingerprint  # noqa: E402
from schema_grounding.selector_labels import (  # noqa: E402
    NEGATIVE_SELECTOR_LABEL,
    POSITIVE_SELECTOR_LABEL,
    format_selector_response,
    selector_label_from_binary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/cypherbench_schema_grounding_full_final"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Positive even batch size used for pre-interleaving; it must match "
            "per_device_train_batch_size. Default: 2."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8").strip()


def format_generator_rows(
    rows: Iterable[dict[str, Any]], system_prompt: str, user_template: str
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        schema = json.dumps(row["sub_schema"], ensure_ascii=False, indent=2)
        prepared.append(
            {
                "task": "generator",
                "system_prompt": system_prompt,
                "user_prompt": user_template.format(question=row["question"], schema=schema),
                "response": json.dumps({"cypher": row["cypher"]}, ensure_ascii=False),
                "example_id": row["id"],
                "source": row["source"],
                "split": row["split"],
                "graph": row["graph"],
            }
        )
    return prepared


def format_selector_rows(
    rows: Iterable[dict[str, Any]], system_prompt: str, user_template: str
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        classification_label = selector_label_from_binary(row["label"])
        prepared.append(
            {
                "task": "selector",
                "system_prompt": system_prompt,
                "user_prompt": user_template.format(
                    question=row["question"], schema_unit=row["unit"]["text"]
                ),
                "response": format_selector_response(classification_label),
                "example_id": row["example_id"],
                "source": row["source"],
                "split": row["split"],
                "graph": row["graph"],
                "schema_id": row["schema_id"],
                "unit_id": row["unit_id"],
                "label": classification_label,
                **(
                    {"contrast_pair_id": str(row["contrast_pair_id"])}
                    if "contrast_pair_id" in row
                    else {}
                ),
            }
        )
    return prepared


def sample_eval_selector_rows(
    rows: list[dict[str, Any]], target: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Choose one balanced selector row per question for task-balanced eval."""

    if target > len({row["example_id"] for row in rows}):
        raise ValueError("Eval selector target exceeds the number of unique questions")
    targets = {
        NEGATIVE_SELECTOR_LABEL: target // 2,
        POSITIVE_SELECTOR_LABEL: target - target // 2,
    }
    chosen: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    for label in (NEGATIVE_SELECTOR_LABEL, POSITIVE_SELECTOR_LABEL):
        candidates = [row for row in rows if row["label"] == label]
        rng.shuffle(candidates)
        per_question: dict[str, dict[str, Any]] = {}
        for row in candidates:
            per_question.setdefault(str(row["example_id"]), row)
        label_rows = [row for question, row in per_question.items() if question not in used_questions]
        if len(label_rows) < targets[label]:
            raise ValueError(f"Not enough unique eval questions for selector label {label}")
        additions = label_rows[: targets[label]]
        chosen.extend(additions)
        used_questions.update(str(row["example_id"]) for row in additions)
    return chosen


def interleave_task_batches(
    generator_rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]], batch_size: int, rng: random.Random
) -> list[dict[str, Any]]:
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("batch-size must be a positive even number")
    if len(generator_rows) != len(selector_rows):
        raise ValueError("Generator and selector row counts must match before interleaving")
    per_task = batch_size // 2
    generators = list(generator_rows)
    selectors = list(selector_rows)
    rng.shuffle(generators)
    rng.shuffle(selectors)
    result: list[dict[str, Any]] = []
    for offset in range(0, len(generators), per_task):
        result.extend(generators[offset : offset + per_task])
        result.extend(selectors[offset : offset + per_task])
    return result


def interleave_without_replacement(
    generator_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Mix tasks while retaining same-question selector contrast pairs.

    Mixed batches are emitted first and later shuffled only in contrast-safe
    blocks by the trainer. Paired selector rows precede unpaired negative rows.
    At batch size two, each contrast pair spans two consecutive mixed batches;
    at larger batch sizes, multiple selector rows share a batch. Remaining
    selector rows form negative-only mixed batches, followed by generator-only
    batches. No source row is duplicated.
    """

    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch-size must be a positive even number")
    if not generator_rows or not selector_rows:
        raise ValueError("Both generator and selector datasets must be non-empty")
    if len(generator_rows) < len(selector_rows):
        raise ValueError("This preparation policy expects generator rows to be at least as numerous as selector rows")

    by_contrast: dict[str, list[dict[str, Any]]] = {}
    unpaired_negatives: list[dict[str, Any]] = []
    for row in selector_rows:
        pair_id = row.get("contrast_pair_id")
        if pair_id is None:
            if row.get("label") != NEGATIVE_SELECTOR_LABEL:
                raise ValueError("Only negative train selector rows may be unpaired")
            unpaired_negatives.append(row)
            continue
        by_contrast.setdefault(str(pair_id), []).append(row)
    malformed = {
        pair_id: [row.get("label") for row in pair]
        for pair_id, pair in by_contrast.items()
        if len(pair) != 2
        or {row.get("label") for row in pair}
        != {NEGATIVE_SELECTOR_LABEL, POSITIVE_SELECTOR_LABEL}
    }
    if malformed:
        raise ValueError(f"Malformed same-question selector contrast pairs: {malformed}")

    contrast_pairs = list(by_contrast.values())
    rng.shuffle(contrast_pairs)
    for pair in contrast_pairs:
        rng.shuffle(pair)
    rng.shuffle(unpaired_negatives)
    selectors = [row for pair in contrast_pairs for row in pair] + unpaired_negatives

    generators = list(generator_rows)
    rng.shuffle(generators)
    generator_offset = 0
    result: list[dict[str, Any]] = []
    selector_capacity = batch_size // 2
    for selector_offset in range(0, len(selectors), selector_capacity):
        selector_chunk = selectors[selector_offset : selector_offset + selector_capacity]
        generator_count = min(
            batch_size - len(selector_chunk),
            len(generators) - generator_offset,
        )
        result.extend(generators[generator_offset : generator_offset + generator_count])
        result.extend(selector_chunk)
        generator_offset += generator_count
    result.extend(generators[generator_offset:])
    if len(result) != len(generator_rows) + len(selector_rows):
        raise AssertionError("Interleaving did not consume every source row exactly once")
    return result


def prepare_output_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input-dir and output-dir must differ")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = [
        "generation_train.jsonl", "generation_dev.jsonl", "generation_test.jsonl",
        "selection_train.jsonl", "selection_dev.jsonl", "selection_test.jsonl",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {', '.join(missing)}")

    generator_system = load_prompt("generator/system_prompt.txt")
    generator_user = load_prompt("generator/user_prompt.txt")
    selector_system = load_prompt("selector/system_prompt.txt")
    selector_user = load_prompt("selector/user_prompt.txt")
    rng = random.Random(args.seed)

    generator_train = format_generator_rows(
        read_jsonl(input_dir / "generation_train.jsonl"), generator_system, generator_user
    )
    selector_train = format_selector_rows(
        read_jsonl(input_dir / "selection_train.jsonl"), selector_system, selector_user
    )
    generator_eval = format_generator_rows(
        read_jsonl(input_dir / "generation_dev.jsonl"), generator_system, generator_user
    )
    selector_eval_full = format_selector_rows(
        read_jsonl(input_dir / "selection_dev.jsonl"), selector_system, selector_user
    )
    generator_test = format_generator_rows(
        read_jsonl(input_dir / "generation_test.jsonl"), generator_system, generator_user
    )
    selector_test = format_selector_rows(
        read_jsonl(input_dir / "selection_test.jsonl"), selector_system, selector_user
    )

    eval_selector_balanced = sample_eval_selector_rows(selector_eval_full, len(generator_eval), rng)
    train = interleave_without_replacement(generator_train, selector_train, args.batch_size, rng)
    evaluation = interleave_task_batches(generator_eval, eval_selector_balanced, args.batch_size, rng)

    prepare_output_dir(input_dir, output_dir, args.overwrite)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "eval.jsonl", evaluation)
    write_jsonl(output_dir / "test_generator.jsonl", generator_test)
    write_jsonl(output_dir / "test_selector.jsonl", selector_test)

    manifest = {
        "input_directory": str(input_dir),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "preparation_fingerprint": preparation_fingerprint(
            input_dir,
            ROOT / "prompts",
            seed=args.seed,
        ),
        "selector_output_format": {
            "type": "json",
            "field": "label",
            "labels": [POSITIVE_SELECTOR_LABEL, NEGATIVE_SELECTOR_LABEL],
        },
        "batch_policy": (
            "Mixed batches begin with adjacent same-question YES/NO selector contrasts, followed by unpaired "
            "negative selector rows; remaining batches contain generator rows only. Source rows are never "
            "duplicated. Keep individual-row dataloader shuffle disabled."
        ),
        "files": {
            "train": {"rows": len(train), "generator": len(generator_train), "selector": len(selector_train)},
            "eval": {
                "rows": len(evaluation),
                "generator": len(generator_eval),
                "selector": len(eval_selector_balanced),
            },
            "test_generator": {"rows": len(generator_test)},
            "test_selector": {"rows": len(selector_test)},
        },
        "eval_selector_labels": dict(Counter(row["label"] for row in eval_selector_balanced)),
        "train_selector_labels": dict(Counter(row["label"] for row in selector_train)),
        "train_selector_contrast_pairs": len(
            {str(row["contrast_pair_id"]) for row in selector_train if "contrast_pair_id" in row}
        ),
        "train_selector_unpaired_negatives": sum(
            1 for row in selector_train if "contrast_pair_id" not in row
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
