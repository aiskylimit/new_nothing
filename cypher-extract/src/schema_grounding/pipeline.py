"""Streaming writer for schema-selection and sub-schema-to-Cypher supervision."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TextIO

from .cypher import SubSchemaExtraction, extract_subschema
from .datasets import SUPPORTED_SOURCES, BenchmarkExample, iter_benchmark_examples

FORMAT_VERSION = "1.2"


@dataclass
class SplitWriters:
    """Open output files associated with one benchmark split."""

    selection: TextIO
    generation: TextIO
    rejected: TextIO
    normalization_issues: TextIO
    inference_selection: TextIO | None = None
    inference_generation: TextIO | None = None

    def close(self) -> None:
        for handle in (
            self.selection,
            self.generation,
            self.rejected,
            self.normalization_issues,
            self.inference_selection,
            self.inference_generation,
        ):
            if handle is not None:
                handle.close()


def _write_jsonl(handle: TextIO, record: dict[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _selection_units(
    example: BenchmarkExample,
    extraction: SubSchemaExtraction,
    *,
    include_labels: bool = True,
) -> list[dict[str, object]]:
    positive_ids = set(extraction.node_unit_ids) | set(extraction.relation_unit_ids)
    rows: list[dict[str, object]] = []
    for unit in example.schema.units:
        unit_dict = unit.to_unit_dict()
        row: dict[str, object] = {
            "id": f"{example.example_id}:{unit.id}",
            "example_id": example.example_id,
            "source": example.source,
            "split": example.split,
            "graph": example.graph,
            "schema_id": example.schema.schema_id,
            "question": example.question,
            "unit_id": unit.id,
            "unit_type": unit_dict["kind"],
            "unit": unit_dict,
        }
        if include_labels:
            row["label"] = int(unit.id in positive_ids)
        rows.append(row)
    return rows


def _example_context(example: BenchmarkExample) -> dict[str, object]:
    return {
        "example_id": example.example_id,
        "source": example.source,
        "split": example.split,
        "graph": example.graph,
        "schema_id": example.schema.schema_id,
        "schema_reference": example.schema_reference,
        "question": example.question,
        "cypher": example.cypher,
    }


def _normalization_issue_record(example: BenchmarkExample) -> dict[str, object]:
    record = _example_context(example)
    record["issues"] = list(example.normalization_issues)
    if example.normalization_error is not None:
        record["error"] = example.normalization_error
    if example.raw_schema is not None:
        record["raw_schema"] = example.raw_schema
    return record


def _sample_rows(
    rows: list[dict[str, object]], negative_ratio: float | None, seed: int
) -> list[dict[str, object]]:
    if negative_ratio is None:
        return rows
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    maximum_negatives = round(len(positives) * negative_ratio)
    if len(negatives) <= maximum_negatives:
        return rows
    # Python's built-in hash is randomized per process; derive a stable seed so
    # negative sampling is reproducible across machines and runs.
    digest = sha256(str(rows[0]["example_id"]).encode("utf-8")).digest()
    example_seed = seed ^ int.from_bytes(digest[:8], byteorder="big")
    sampler = random.Random(example_seed)
    selected = sampler.sample(negatives, maximum_negatives)
    return sorted([*positives, *selected], key=lambda row: str(row["unit_id"]))


def _rejection_reason(extraction: SubSchemaExtraction, strict: bool) -> str | None:
    if not extraction.has_units:
        return "no_schema_units_resolved"
    if strict and not extraction.complete:
        return "incomplete_gold_subschema"
    return None


def _validate_output_paths(output_dir: Path, splits: Iterable[str], overwrite: bool) -> None:
    names = ["schemas.jsonl", "manifest.json"]
    for split in splits:
        names.extend(
            [
                f"selection_{split}.jsonl",
                f"generation_{split}.jsonl",
                f"rejected_{split}.jsonl",
                f"normalization_issues_{split}.jsonl",
            ]
        )
        if split == "test":
            names.extend(
                [
                    "selection_inference_test.jsonl",
                    "generation_inference_test.jsonl",
                ]
            )
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite output files: {rendered}. Use --overwrite.")


def build_dataset(
    benchmarks_root: Path,
    output_dir: Path,
    sources: Iterable[str] = SUPPORTED_SOURCES,
    splits: Iterable[str] = ("train",),
    *,
    strict: bool = True,
    negative_ratio: float | None = None,
    max_examples: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build JSONL training data and return the reproducibility manifest.

    In strict mode the main corpora contain only examples for which every required
    node/relation pattern maps to the provided full schema. Broad relationship
    patterns retain every matching relation unit. Rejected records are retained
    with diagnostics for audit rather than being discarded.
    """

    sources = tuple(sources)
    splits = tuple(splits)
    unknown_sources = set(sources) - set(SUPPORTED_SOURCES)
    if unknown_sources:
        raise ValueError(f"Unknown sources: {sorted(unknown_sources)}")
    if negative_ratio is not None and negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_output_paths(output_dir, splits, overwrite)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    emitted_schema_ids: set[str] = set()

    with (output_dir / "schemas.jsonl").open("w", encoding="utf-8") as schemas_handle:
        split_handles: dict[str, SplitWriters] = {}
        try:
            for split in splits:
                split_handles[split] = SplitWriters(
                    selection=(output_dir / f"selection_{split}.jsonl").open(
                        "w", encoding="utf-8"
                    ),
                    generation=(output_dir / f"generation_{split}.jsonl").open(
                        "w", encoding="utf-8"
                    ),
                    rejected=(output_dir / f"rejected_{split}.jsonl").open(
                        "w", encoding="utf-8"
                    ),
                    normalization_issues=(
                        output_dir / f"normalization_issues_{split}.jsonl"
                    ).open("w", encoding="utf-8"),
                    inference_selection=(
                        (output_dir / "selection_inference_test.jsonl").open(
                            "w", encoding="utf-8"
                        )
                        if split == "test"
                        else None
                    ),
                    inference_generation=(
                        (output_dir / "generation_inference_test.jsonl").open(
                            "w", encoding="utf-8"
                        )
                        if split == "test"
                        else None
                    ),
                )

            for source in sources:
                for split in splits:
                    key = f"{source}/{split}"
                    writers = split_handles[split]
                    for index, example in enumerate(
                        iter_benchmark_examples(benchmarks_root, source, split)
                    ):
                        if max_examples is not None and index >= max_examples:
                            break
                        counts[key]["records_seen"] += 1
                        if example.normalization_issues:
                            counts[key]["normalization_issue_records"] += 1
                            for issue in example.normalization_issues:
                                counts[key][f"normalization_issue_{issue}"] += 1
                            _write_jsonl(
                                writers.normalization_issues,
                                _normalization_issue_record(example),
                            )
                        if not example.question or not example.cypher:
                            counts[key]["records_missing_question_or_cypher"] += 1
                            _write_jsonl(
                                writers.rejected,
                                {
                                    **_example_context(example),
                                    "reason": "missing_question_or_cypher",
                                },
                            )
                            continue
                        if not example.schema.units:
                            counts[key]["records_empty_schema"] += 1
                            _write_jsonl(
                                writers.rejected,
                                {
                                    **_example_context(example),
                                    "reason": "empty_normalized_schema",
                                },
                            )
                            continue

                        # Keep the normalized full schema even when the example
                        # itself is rejected; rejected.jsonl references this ID
                        # and can therefore be audited without reparsing input.
                        if example.schema.schema_id not in emitted_schema_ids:
                            _write_jsonl(schemas_handle, example.schema.to_dict())
                            emitted_schema_ids.add(example.schema.schema_id)
                            counts[key]["schemas_emitted"] += 1

                        extraction = extract_subschema(example.cypher, example.schema)
                        counts[key]["unmapped_node_labels"] += len(extraction.unmapped_node_labels)
                        counts[key]["unmapped_relation_types"] += len(
                            extraction.unmapped_relation_types
                        )
                        counts[key]["unmatched_relationship_patterns"] += len(
                            extraction.unmatched_relationship_patterns
                        )
                        counts[key]["unresolved_node_patterns"] += extraction.unresolved_node_patterns
                        counts[key]["ambiguous_relation_patterns"] += extraction.ambiguous_relation_patterns

                        reason = _rejection_reason(extraction, strict)

                        # Test inference must cover every example that has a usable
                        # question, reference Cypher, and full schema. Gold extraction
                        # completeness only controls whether selector labels and a
                        # reference sub-schema are trustworthy; it must not decide
                        # whether the end-to-end model gets evaluated.
                        if writers.inference_generation is not None:
                            gold_subschema_available = extraction.has_units and extraction.complete
                            inference_generation: dict[str, object] = {
                                "id": example.example_id,
                                "source": example.source,
                                "split": example.split,
                                "graph": example.graph,
                                "schema_id": example.schema.schema_id,
                                "question": example.question,
                                "cypher": example.cypher,
                                "gold_subschema_available": gold_subschema_available,
                            }
                            if gold_subschema_available:
                                inference_generation["sub_schema"] = example.schema.subset_dict(
                                    extraction.node_unit_ids,
                                    extraction.relation_unit_ids,
                                )
                            else:
                                inference_generation["gold_subschema_diagnostics"] = (
                                    extraction.diagnostics()
                                )
                            _write_jsonl(writers.inference_generation, inference_generation)

                            inference_selection_rows = _selection_units(
                                example,
                                extraction,
                                include_labels=gold_subschema_available,
                            )
                            assert writers.inference_selection is not None
                            for row in inference_selection_rows:
                                _write_jsonl(writers.inference_selection, row)
                            counts[key]["inference_examples"] += 1
                            counts[key]["inference_selection_examples"] += len(
                                inference_selection_rows
                            )
                            label_status = "labeled" if gold_subschema_available else "unlabeled"
                            counts[key][f"inference_{label_status}_examples"] += 1
                            counts[key][f"inference_selection_{label_status}"] += len(
                                inference_selection_rows
                            )

                        if reason:
                            counts[key]["records_rejected"] += 1
                            _write_jsonl(
                                writers.rejected,
                                {
                                    **_example_context(example),
                                    "reason": reason,
                                    "diagnostics": extraction.diagnostics(),
                                },
                            )
                            continue

                        sub_schema = example.schema.subset_dict(
                            extraction.node_unit_ids, extraction.relation_unit_ids
                        )
                        _write_jsonl(
                            writers.generation,
                            {
                                "id": example.example_id,
                                "source": example.source,
                                "split": example.split,
                                "graph": example.graph,
                                "schema_id": example.schema.schema_id,
                                "question": example.question,
                                "sub_schema": sub_schema,
                                "cypher": example.cypher,
                            },
                        )
                        counts[key]["generation_examples"] += 1

                        selection_rows = _sample_rows(
                            _selection_units(example, extraction), negative_ratio, seed
                        )
                        for row in selection_rows:
                            _write_jsonl(writers.selection, row)
                        counts[key]["selection_examples"] += len(selection_rows)
                        counts[key]["selection_positive"] += sum(
                            int(row["label"]) for row in selection_rows
                        )
                        counts[key]["selection_negative"] += sum(
                            1 - int(row["label"]) for row in selection_rows
                        )
                    if counts[key]["records_seen"] and not counts[key]["generation_examples"]:
                        raise ValueError(
                            f"{key} produced no usable generation examples from "
                            f"{counts[key]['records_seen']} input records."
                        )
        finally:
            for writers in split_handles.values():
                writers.close()

    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "benchmarks_root": str(benchmarks_root),
        "sources": list(sources),
        "splits": list(splits),
        "strict": strict,
        "negative_ratio": negative_ratio,
        "max_examples_per_source_split": max_examples,
        "seed": seed,
        "unique_schemas": len(emitted_schema_ids),
        "counts": {key: dict(sorted(value.items())) for key, value in sorted(counts.items())},
        "files": {
            "schemas": "schemas.jsonl",
            "selection": {split: f"selection_{split}.jsonl" for split in splits},
            "generation": {split: f"generation_{split}.jsonl" for split in splits},
            "rejected": {split: f"rejected_{split}.jsonl" for split in splits},
            "normalization_issues": {
                split: f"normalization_issues_{split}.jsonl" for split in splits
            },
            "inference_selection": (
                "selection_inference_test.jsonl" if "test" in splits else None
            ),
            "inference_generation": (
                "generation_inference_test.jsonl" if "test" in splits else None
            ),
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
