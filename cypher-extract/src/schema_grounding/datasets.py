"""Adapters that expose the structured training records of each benchmark."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    CanonicalSchema,
    canonical_schema,
    from_cypherbench,
    from_mind_the_query,
    from_neo4j_schema_text,
)

SUPPORTED_SOURCES = ("cypherbench", "mind_the_query", "neo4j_text2cypher")


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    source: str
    split: str
    graph: str
    question: str
    cypher: str
    schema: CanonicalSchema
    schema_reference: str
    raw_schema: str | None = None
    normalization_issues: tuple[str, ...] = ()
    normalization_error: str | None = None


def _load_json(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object on line {line_number} in {path}")
                records.append(record)
        return records
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _parse_cypherbench_prompt_record(record: Mapping[str, Any]) -> tuple[str, dict[str, Any], str]:
    """Extract question, schema, and gold Cypher from CypherBench prompt JSONL."""

    prompt = str(record.get("user_prompt", ""))
    question_prefix = "QUESTION:\n"
    schema_prefix = "\n\nSCHEMA:\n"
    generation_prefix = "\n\nGenerate a Cypher query"
    if not prompt.startswith(question_prefix) or schema_prefix not in prompt:
        raise ValueError("Unsupported CypherBench prompt record format")
    question, remainder = prompt[len(question_prefix) :].split(schema_prefix, 1)
    schema_text = remainder.split(generation_prefix, 1)[0]
    schema_payload = json.loads(schema_text)
    if not isinstance(schema_payload, dict):
        raise ValueError("Expected CypherBench prompt schema to be a JSON object")
    response = json.loads(str(record.get("response", "")))
    if not isinstance(response, dict) or not isinstance(response.get("cypher"), str):
        raise ValueError("Expected CypherBench prompt response to contain a Cypher string")
    return question, schema_payload, response["cypher"]


def _split_file(root: Path, source_dir: str, split: str) -> Path:
    mapping = {
        "train": ("train.jsonl", "train.json"),
        "dev": ("dev.jsonl", "dev.json"),
        "val": ("dev.jsonl", "dev.json"),
        "test": ("test.jsonl", "test.json"),
    }
    if source_dir in {"Mind_the_query", "Neo4j_Text2Cypher"}:
        # These JSONL files are prompt/response exports without usable graph
        # metadata. The structured JSON files retain the schema needed for
        # canonicalization and gold sub-schema extraction.
        mapping = {
            "train": ("train.json",),
            "dev": ("dev.json",),
            "val": ("dev.json",),
            "test": ("test.json",),
        }
    if source_dir == "Mind_the_query" and split == "train":
        mapping["train"] = ("train.json", "train_val.json")
    if split not in mapping:
        raise ValueError(
            f"{source_dir} has no structured '{split}' split. Supported splits: train, dev (or val), test."
        )
    for filename in mapping[split]:
        path = root / source_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(root / source_dir / mapping[split][0])


def _normalise_graph(value: Any) -> str:
    return str(value) if value not in (None, "") else "unknown"


def _load_normalized_schema(
    cache: dict[str, tuple[CanonicalSchema, tuple[str, ...], str | None]],
    cache_key: str,
    source: str,
    graph: str,
    loader: Callable[[], CanonicalSchema],
) -> tuple[CanonicalSchema, tuple[str, ...], str | None]:
    """Load a schema once, retaining normalization failures for later audit."""

    if cache_key in cache:
        return cache[cache_key]

    try:
        schema = loader()
    except Exception as error:  # malformed released schema must not abort a corpus build
        result = (
            canonical_schema(source, graph, (), ()),
            ("normalization_error",),
            f"{type(error).__name__}: {error}",
        )
    else:
        issues = () if schema.units else ("empty_normalized_schema",)
        result = (schema, issues, None)
    cache[cache_key] = result
    return result


def iter_benchmark_examples(
    benchmarks_root: Path, source: str, split: str
) -> Iterator[BenchmarkExample]:
    """Yield examples with a normalized schema, retaining their original split."""

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source '{source}'. Expected one of {SUPPORTED_SOURCES}.")

    directory = {
        "cypherbench": "Cypherbench",
        "mind_the_query": "Mind_the_query",
        "neo4j_text2cypher": "Neo4j_Text2Cypher",
    }[source]
    records_path = _split_file(benchmarks_root, directory, split)
    records = _load_json(records_path)
    schema_cache: dict[str, tuple[CanonicalSchema, tuple[str, ...], str | None]] = {}

    for index, record in enumerate(records):
        graph = _normalise_graph(record.get("graph"))
        raw_schema: str | None = None
        cypher = str(record.get("gold_cypher", ""))
        if source == "cypherbench":
            if "user_prompt" in record:
                question, schema_payload, cypher = _parse_cypherbench_prompt_record(record)
                graph = _normalise_graph(schema_payload.get("name"))
                cache_key = graph
                schema_reference = f"{records_path.name}[{index}].user_prompt.SCHEMA"
                def schema_loader() -> CanonicalSchema:
                    return from_cypherbench(schema_payload, graph)
            else:
                cache_key = graph
                schema_path = benchmarks_root / directory / "graphs" / "schemas" / f"{graph}_schema.json"
                question = str(record.get("nl_question", ""))
                schema_reference = str(schema_path.relative_to(benchmarks_root))
                def schema_loader() -> CanonicalSchema:
                    return from_cypherbench(_load_json_object(schema_path), graph)
            schema, issues, normalization_error = _load_normalized_schema(
                schema_cache,
                cache_key,
                source,
                graph,
                schema_loader,
            )
        elif source == "mind_the_query":
            cache_key = graph
            schema_path = benchmarks_root / directory / "graphs" / "schemas" / f"{graph}.json"
            schema, issues, normalization_error = _load_normalized_schema(
                schema_cache,
                cache_key,
                source,
                graph,
                lambda: from_mind_the_query(_load_json_object(schema_path), graph),
            )
            question = str(record.get("question", record.get("nl_question", "")))
            schema_reference = str(schema_path.relative_to(benchmarks_root))
        else:
            raw_schema = str(record.get("schema", ""))
            # Identical text can occur under different graph names. The graph
            # remains part of a canonical schema ID, so it must also scope the
            # cache entry.
            cache_key = f"{graph}\0{raw_schema}"
            schema, issues, normalization_error = _load_normalized_schema(
                schema_cache,
                cache_key,
                source,
                graph,
                lambda: from_neo4j_schema_text(raw_schema, graph),
            )
            question = str(record.get("nl_question", record.get("question", "")))
            schema_reference = f"{records_path.name}[{index}].schema"

        example_key = record.get("qid", record.get("instance_id", index))
        yield BenchmarkExample(
            # Mind-the-Query reuses numeric qids across graphs, so graph must
            # be part of the corpus key. Including it for every source also
            # makes IDs self-describing when datasets are mixed.
            example_id=f"{source}:{split}:{graph}:{example_key}",
            source=source,
            split=split,
            graph=graph,
            question=question.strip(),
            cypher=cypher.strip(),
            schema=schema,
            schema_reference=schema_reference,
            raw_schema=raw_schema,
            normalization_issues=issues,
            normalization_error=normalization_error,
        )
