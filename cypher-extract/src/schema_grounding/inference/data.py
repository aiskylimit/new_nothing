from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: Path
    generation_filename: str = "generation_test.jsonl"
    selection_filename: str = "selection_test.jsonl"

    @property
    def generation_test(self) -> Path:
        return self.directory / self.generation_filename

    @property
    def selection_test(self) -> Path:
        return self.directory / self.selection_filename


def default_dataset_specs(repository_root: Path) -> dict[str, DatasetSpec]:
    data = repository_root / "data"
    inference_files = {
        "generation_filename": "generation_inference_test.jsonl",
        "selection_filename": "selection_inference_test.jsonl",
    }
    return {
        "cypherbench": DatasetSpec(
            "cypherbench", data / "cypherbench_schema_grounding_full_final", **inference_files
        ),
        "mind_the_query": DatasetSpec(
            "mind_the_query", data / "mind_the_query_schema_grounding_full", **inference_files
        ),
        "neo4j_text2cypher": DatasetSpec(
            "neo4j_text2cypher",
            data / "neo4j_text2cypher_schema_grounding_full",
            **inference_files,
        ),
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield row


def load_generation_index(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"Generation row in {path} has no id")
        if example_id in rows:
            raise ValueError(f"Duplicate generation id {example_id!r} in {path}")
        order.append(example_id)
        rows[example_id] = row
    if not rows:
        raise ValueError(f"No generation rows found in {path}")
    return order, rows


def paired_rows(left_path: Path, right_path: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    sentinel = object()
    pairs = zip_longest(iter_jsonl(left_path), iter_jsonl(right_path), fillvalue=sentinel)
    for index, (left, right) in enumerate(pairs, 1):
        if left is sentinel or right is sentinel:
            raise ValueError(f"Row-count mismatch between {left_path} and {right_path} at row {index}")
        assert isinstance(left, dict) and isinstance(right, dict)
        yield left, right


def grouped_paired_units(
    selection_path: Path, prediction_path: Path
) -> Iterator[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    current_example: str | None = None
    source_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    finished: set[str] = set()
    for source, prediction in paired_rows(selection_path, prediction_path):
        if source.get("id") != prediction.get("id"):
            raise ValueError(f"Selector prediction misalignment: {source.get('id')!r} != {prediction.get('id')!r}")
        example_id = source.get("example_id")
        if not isinstance(example_id, str):
            raise ValueError(f"Selector row {source.get('id')!r} has no example_id")
        if current_example is None:
            current_example = example_id
        if example_id != current_example:
            finished.add(current_example)
            yield current_example, source_rows, prediction_rows
            if example_id in finished:
                raise ValueError(f"Selector rows for {example_id!r} are not contiguous")
            current_example = example_id
            source_rows = []
            prediction_rows = []
        source_rows.append(source)
        prediction_rows.append(prediction)
    if current_example is not None:
        yield current_example, source_rows, prediction_rows
