from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from distillation.generation import selector_generation_kwargs
from distillation.metrics import compute_task_metrics, extract_cypher
from schema_grounding.inference.checkpoints import LastCheckpoint
from schema_grounding.inference.data import (
    DatasetSpec,
    grouped_paired_units,
    iter_jsonl,
    load_generation_index,
    paired_rows,
)
from schema_grounding.inference.merge import merge_schema_units
from schema_grounding.inference.outputs import ResumableJsonl, write_json_atomic, write_json_line
from schema_grounding.inference.parsing import parse_selector_label
from schema_grounding.inference.prompting import (
    Message,
    PromptTemplates,
    chat_template_metadata,
)
from schema_grounding.selector_labels import (
    NEGATIVE_SELECTOR_LABEL,
    POSITIVE_SELECTOR_LABEL,
    SELECTOR_LABELS,
    selector_label_from_binary,
)


class GenerationRunner(Protocol):
    model: Any

    def generate(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
    ) -> list[str]: ...

    def prompt_length(self, messages: Sequence[Message]) -> int: ...


@dataclass(frozen=True)
class InferenceOptions:
    selector_batch_size: int = 128
    generator_batch_size: int = 16
    selector_max_new_tokens: int = 16
    generator_max_new_tokens: int = 256
    close_relation_endpoints: bool = True
    generator_do_sample: bool = True
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 0
    num_beams: int = 1
    seed: int = 42

    def validate(self) -> None:
        for name in (
            "selector_batch_size",
            "generator_batch_size",
            "selector_max_new_tokens",
            "generator_max_new_tokens",
            "num_beams",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def _generator_generation_kwargs(options: InferenceOptions) -> dict[str, Any]:
    return {
        "do_sample": options.generator_do_sample,
        "temperature": options.temperature,
        "top_p": options.top_p,
        "top_k": options.top_k,
        "num_beams": options.num_beams,
    }


def _batches(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_after_progress(
    source_path: Path, completed_rows: int, last_completed_id: str | None
) -> Iterator[dict[str, Any]]:
    observed = 0
    for observed, row in enumerate(iter_jsonl(source_path), 1):
        if observed <= completed_rows:
            if observed == completed_rows and row.get("id") != last_completed_id:
                raise ValueError(
                    f"Resume mismatch at row {observed}: source has {row.get('id')!r}, "
                    f"partial output has {last_completed_id!r}"
                )
            continue
        yield row
    if completed_rows > observed:
        raise ValueError(f"Partial output has {completed_rows} rows but source {source_path} has only {observed}")


def run_selector_stage(
    spec: DatasetSpec,
    runner: GenerationRunner | None,
    templates: PromptTemplates,
    output_path: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    output = ResumableJsonl(output_path)
    if output.complete:
        output.rng_state_path.unlink(missing_ok=True)
        output.rng_temporary_path.unlink(missing_ok=True)
        return {"status": "reused", "path": str(output_path)}
    if runner is None:
        raise RuntimeError("Selector output is incomplete but no model runner was loaded")
    completed_rows, last_id = output.restore_rng_progress()
    started = time.monotonic()
    generated_rows = 0
    invalid_rows = 0
    with output.open_append() as handle:
        source = _rows_after_progress(spec.selection_test, completed_rows, last_id)
        for batch in _batches(source, options.selector_batch_size):
            conversations = [
                templates.selector_messages(str(row["question"]), str(row["unit"]["text"])) for row in batch
            ]
            selector_kwargs = selector_generation_kwargs()
            selector_kwargs["max_new_tokens"] = options.selector_max_new_tokens
            raw_outputs = runner.generate(conversations, **selector_kwargs)
            if len(raw_outputs) != len(batch):
                raise RuntimeError("Model returned the wrong number of selector outputs")
            for row, raw_output in zip(batch, raw_outputs, strict=True):
                parsed = parse_selector_label(raw_output)
                invalid_rows += parsed is None
                write_json_line(
                    handle,
                    {
                        "id": row["id"],
                        "example_id": row["example_id"],
                        "schema_id": row["schema_id"],
                        "unit_id": row["unit_id"],
                        "unit_type": row["unit_type"],
                        "predicted_label": parsed or "INVALID",
                        "valid": parsed is not None,
                        "raw_output": raw_output,
                    },
                )
                generated_rows += 1
            output.checkpoint_rng_progress(
                handle,
                completed_rows + generated_rows,
                str(batch[-1]["id"]),
            )
            if generated_rows and generated_rows % (options.selector_batch_size * 100) == 0:
                print(f"[{spec.name}/selector] generated {completed_rows + generated_rows} unit labels", flush=True)
    output.publish()
    return {
        "status": "completed",
        "path": str(output_path),
        "resumed_rows": completed_rows,
        "generated_rows": generated_rows,
        "invalid_rows_in_current_run": invalid_rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_merge_stage(
    spec: DatasetSpec,
    selector_predictions: Path,
    output_path: Path,
    *,
    close_relation_endpoints: bool,
) -> dict[str, Any]:
    if output_path.is_file():
        return {"status": "reused", "path": str(output_path)}
    generation_order, generation_rows = load_generation_index(spec.generation_test)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_examples: set[str] = set()
    closure_count = 0
    empty_count = 0
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example_id, source_rows, prediction_rows in grouped_paired_units(
            spec.selection_test, selector_predictions
        ):
            generation = generation_rows.get(example_id)
            if generation is None:
                raise ValueError(f"Selector example {example_id!r} has no generator test row")
            units = [row["unit"] for row in source_rows]
            related_ids = [
                str(source["unit_id"])
                for source, prediction in zip(source_rows, prediction_rows, strict=True)
                if prediction["predicted_label"] == POSITIVE_SELECTOR_LABEL
            ]
            merged = merge_schema_units(
                units,
                related_ids,
                close_relation_endpoints=close_relation_endpoints,
            )
            closure_count += len(merged.closure_added_node_ids)
            empty_count += not merged.sub_schema["nodes"] and not merged.sub_schema["relationships"]
            write_json_line(
                handle,
                {
                    "id": example_id,
                    "source": generation["source"],
                    "split": generation["split"],
                    "graph": generation["graph"],
                    "schema_id": generation["schema_id"],
                    "question": generation["question"],
                    "predicted_sub_schema": merged.sub_schema,
                    "directly_selected_unit_ids": list(merged.directly_selected_unit_ids),
                    "closure_added_node_ids": list(merged.closure_added_node_ids),
                },
            )
            seen_examples.add(example_id)
    missing = [example_id for example_id in generation_order if example_id not in seen_examples]
    if missing:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Generator examples have no selector units: {missing[:5]}")
    temporary.replace(output_path)
    return {
        "status": "completed",
        "path": str(output_path),
        "examples": len(seen_examples),
        "closure_added_nodes": closure_count,
        "empty_sub_schemas": empty_count,
        "elapsed_seconds": time.monotonic() - started,
    }


def _context_limit(runner: GenerationRunner) -> int | None:
    model_limit = getattr(getattr(runner, "model", None), "config", None)
    value = getattr(model_limit, "max_position_embeddings", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def run_generator_stage(
    spec: DatasetSpec,
    runner: GenerationRunner | None,
    templates: PromptTemplates,
    sub_schema_path: Path,
    output_path: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    output = ResumableJsonl(output_path)
    if output.complete:
        output.rng_state_path.unlink(missing_ok=True)
        output.rng_temporary_path.unlink(missing_ok=True)
        return {"status": "reused", "path": str(output_path)}
    if runner is None:
        raise RuntimeError("Generator output is incomplete but no model runner was loaded")
    _, generation_rows = load_generation_index(spec.generation_test)
    completed_rows, last_id = output.restore_rng_progress()
    started = time.monotonic()
    generated_rows = 0
    max_prompt_length = 0
    context_limit = _context_limit(runner)
    with output.open_append() as handle:
        source = _rows_after_progress(sub_schema_path, completed_rows, last_id)
        for batch in _batches(source, options.generator_batch_size):
            conversations = [
                templates.generator_messages(str(row["question"]), row["predicted_sub_schema"]) for row in batch
            ]
            prompt_lengths = [runner.prompt_length(messages) for messages in conversations]
            max_prompt_length = max(max_prompt_length, *prompt_lengths)
            if context_limit is not None:
                overflowing = [
                    length for length in prompt_lengths if length + options.generator_max_new_tokens > context_limit
                ]
                if overflowing:
                    raise ValueError(
                        f"Generator prompt plus response budget exceeds context limit {context_limit}: "
                        f"prompt_length={max(overflowing)}"
                    )
            raw_outputs = runner.generate(
                conversations,
                max_new_tokens=options.generator_max_new_tokens,
                **_generator_generation_kwargs(options),
            )
            if len(raw_outputs) != len(batch):
                raise RuntimeError("Model returned the wrong number of generator outputs")
            for row, raw_output, prompt_length in zip(batch, raw_outputs, prompt_lengths, strict=True):
                # Gold data is attached only after generation and is never part of the prompt.
                reference = generation_rows[str(row["id"])]
                prediction = {
                    **row,
                    "raw_output": raw_output,
                    "predicted_cypher": extract_cypher(raw_output),
                    "prompt_length": prompt_length,
                    "reference_cypher": reference["cypher"],
                    "gold_subschema_available": bool(
                        reference.get("gold_subschema_available", "sub_schema" in reference)
                    ),
                }
                if "sub_schema" in reference:
                    prediction["reference_sub_schema"] = reference["sub_schema"]
                write_json_line(handle, prediction)
                generated_rows += 1
            output.checkpoint_rng_progress(
                handle,
                completed_rows + generated_rows,
                str(batch[-1]["id"]),
            )
            if generated_rows and generated_rows % (options.generator_batch_size * 25) == 0:
                print(
                    f"[{spec.name}/generator] generated {completed_rows + generated_rows} Cypher queries",
                    flush=True,
                )
    output.publish()
    return {
        "status": "completed",
        "path": str(output_path),
        "resumed_rows": completed_rows,
        "generated_rows": generated_rows,
        "max_prompt_length_in_current_run": max_prompt_length,
        "elapsed_seconds": time.monotonic() - started,
    }


def compute_inference_metrics(
    spec: DatasetSpec,
    selector_predictions: Path,
    generator_predictions: Path,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = invalid = 0
    selector_count = 0
    selector_inference_count = 0
    selector_inference_invalid = 0
    for source, prediction in paired_rows(spec.selection_test, selector_predictions):
        if source.get("id") != prediction.get("id"):
            raise ValueError(
                f"Selector prediction misalignment: {source.get('id')!r} != {prediction.get('id')!r}"
            )
        predicted = prediction["predicted_label"]
        if predicted not in {*SELECTOR_LABELS, "INVALID"}:
            raise ValueError(f"Unknown selector prediction label {predicted!r} for row {prediction.get('id')!r}")
        selector_inference_count += 1
        selector_inference_invalid += predicted == "INVALID"
        if "label" not in source:
            continue
        selector_count += 1
        expected = selector_label_from_binary(source["label"])
        if predicted == "INVALID":
            invalid += 1
            if expected == POSITIVE_SELECTOR_LABEL:
                false_negative += 1
            continue
        if expected == POSITIVE_SELECTOR_LABEL and predicted == POSITIVE_SELECTOR_LABEL:
            true_positive += 1
        elif expected == NEGATIVE_SELECTOR_LABEL and predicted == POSITIVE_SELECTOR_LABEL:
            false_positive += 1
        elif expected == POSITIVE_SELECTOR_LABEL:
            false_negative += 1
        else:
            true_negative += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if selector_inference_count == 0:
        raise ValueError(f"No selector rows found in {spec.selection_test}")

    generation_order, generation_rows = load_generation_index(spec.generation_test)
    predicted_cypher: list[str] = []
    reference_cypher: list[str] = []
    closure_nodes = empty_schemas = 0
    seen_generator_ids: set[str] = set()
    for row in iter_jsonl(generator_predictions):
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"Generator prediction in {generator_predictions} has no id")
        if example_id in seen_generator_ids:
            raise ValueError(f"Duplicate generator prediction id {example_id!r} in {generator_predictions}")
        reference = generation_rows.get(example_id)
        if reference is None:
            raise ValueError(f"Generator prediction id {example_id!r} is not present in {spec.generation_test}")
        seen_generator_ids.add(example_id)
        predicted_cypher.append(str(row["predicted_cypher"]))
        reference_cypher.append(str(reference["cypher"]))
        closure_nodes += len(row["closure_added_node_ids"])
        schema = row["predicted_sub_schema"]
        empty_schemas += not schema["nodes"] and not schema["relationships"]
    missing_generator_ids = [example_id for example_id in generation_order if example_id not in seen_generator_ids]
    if missing_generator_ids:
        raise ValueError(f"Missing generator predictions for ids: {missing_generator_ids[:5]}")
    generator_metrics = compute_task_metrics(predicted_cypher, reference_cypher)
    selector_labeled_examples = sum(
        bool(row.get("gold_subschema_available", True))
        for row in generation_rows.values()
    )
    selector_total_examples = len(generation_rows)
    selector_coverage = selector_labeled_examples / selector_total_examples
    strict_accuracy = (
        100.0 * (true_positive + true_negative) / selector_count
        if selector_count
        else 0.0
    )
    strict_precision = 100.0 * precision
    strict_recall = 100.0 * recall
    strict_f1 = 100.0 * f1
    return {
        "selector": {
            "count": selector_count,
            "inference_count": selector_inference_count,
            "unlabeled_count": selector_inference_count - selector_count,
            "example_count": selector_total_examples,
            "labeled_example_count": selector_labeled_examples,
            "zero_scored_example_count": selector_total_examples
            - selector_labeled_examples,
            "coverage": 100.0 * selector_coverage,
            "accuracy": strict_accuracy * selector_coverage,
            "precision": strict_precision * selector_coverage,
            "recall": strict_recall * selector_coverage,
            "f1": strict_f1 * selector_coverage,
            "strict_accuracy": strict_accuracy,
            "strict_precision": strict_precision,
            "strict_recall": strict_recall,
            "strict_f1": strict_f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "invalid": invalid,
            "inference_invalid": selector_inference_invalid,
        },
        "generator": generator_metrics,
        "pipeline": {
            "closure_added_nodes": closure_nodes,
            "empty_sub_schemas": empty_schemas,
        },
    }


def prepare_run_directory(
    *,
    method: str,
    checkpoint: LastCheckpoint,
    spec: DatasetSpec,
    templates: PromptTemplates,
    output_directory: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    """Validate the resumable run configuration before loading a model."""

    options.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    selector_path = output_directory / "selector_predictions.jsonl"
    selector_partial_path = selector_path.with_suffix(selector_path.suffix + ".partial")
    selector_rng_state_path = selector_path.with_suffix(selector_path.suffix + ".rng_state.pt")
    selector_rng_temporary_path = selector_rng_state_path.with_suffix(selector_rng_state_path.suffix + ".tmp")
    sub_schema_path = output_directory / "predicted_subschemas.jsonl"
    sub_schema_temporary_path = sub_schema_path.with_suffix(sub_schema_path.suffix + ".tmp")
    generator_path = output_directory / "generator_predictions.jsonl"
    generator_partial_path = generator_path.with_suffix(generator_path.suffix + ".partial")
    generator_rng_state_path = generator_path.with_suffix(generator_path.suffix + ".rng_state.pt")
    generator_rng_temporary_path = generator_rng_state_path.with_suffix(generator_rng_state_path.suffix + ".tmp")
    run_config_path = output_directory / "run_config.json"
    inference_artifacts = (
        selector_path,
        selector_partial_path,
        selector_rng_state_path,
        selector_rng_temporary_path,
        sub_schema_path,
        sub_schema_temporary_path,
        generator_path,
        generator_partial_path,
        generator_rng_state_path,
        generator_rng_temporary_path,
        output_directory / "metrics.json",
        output_directory / "manifest.json",
    )
    if not run_config_path.is_file() and any(path.exists() for path in inference_artifacts):
        raise ValueError(
            f"Inference outputs exist in {output_directory} without run_config.json; use a new --output-dir or "
            "remove the orphaned outputs"
        )
    if (sub_schema_path.is_file() or sub_schema_temporary_path.is_file()) and not selector_path.is_file():
        raise ValueError(
            f"Inconsistent inference outputs in {output_directory}: sub-schema output exists without completed "
            "selector_predictions.jsonl"
        )
    if (
        generator_path.is_file() or generator_partial_path.is_file() or generator_rng_state_path.is_file()
    ) and not sub_schema_path.is_file():
        raise ValueError(
            f"Inconsistent inference outputs in {output_directory}: generator output exists without completed "
            "predicted_subschemas.jsonl"
        )
    run_config = {
        "method": method,
        "dataset": spec.name,
        "checkpoint": asdict(checkpoint),
        "options": asdict(options),
        "selection_test": str(spec.selection_test.resolve()),
        "generation_test": str(spec.generation_test.resolve()),
        "input_fingerprints": {
            "selection_test": _file_sha256(spec.selection_test),
            "generation_test": _file_sha256(spec.generation_test),
        },
        "prompt_fingerprints": templates.fingerprints(),
        "chat_template": chat_template_metadata(checkpoint.model_family),
        "selector_protocol": {
            "positive_label": POSITIVE_SELECTOR_LABEL,
            "negative_label": NEGATIVE_SELECTOR_LABEL,
            "output_format": {"label": "YES|NO"},
            "do_sample": False,
            "num_beams": 1,
        },
    }
    if run_config_path.is_file():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError(
                f"Existing inference outputs in {output_directory} were created with a different configuration "
                "or with different input/prompt contents. Choose a new --output-dir or remove that method/dataset "
                "directory."
            )
    else:
        write_json_atomic(run_config_path, run_config)
    return run_config


def model_runner_required(output_directory: Path) -> bool:
    """Return whether either model-backed stage still needs generation."""

    return not (output_directory / "selector_predictions.jsonl").is_file() or not (
        output_directory / "generator_predictions.jsonl"
    ).is_file()


def run_dataset_pipeline(
    *,
    method: str,
    checkpoint: LastCheckpoint,
    spec: DatasetSpec,
    runner: GenerationRunner | None,
    templates: PromptTemplates,
    output_directory: Path,
    options: InferenceOptions,
) -> dict[str, Any]:
    prepare_run_directory(
        method=method,
        checkpoint=checkpoint,
        spec=spec,
        templates=templates,
        output_directory=output_directory,
        options=options,
    )
    selector_path = output_directory / "selector_predictions.jsonl"
    sub_schema_path = output_directory / "predicted_subschemas.jsonl"
    generator_path = output_directory / "generator_predictions.jsonl"
    stages = {
        "selector": run_selector_stage(spec, runner, templates, selector_path, options),
        "merge": run_merge_stage(
            spec,
            selector_path,
            sub_schema_path,
            close_relation_endpoints=options.close_relation_endpoints,
        ),
        "generator": run_generator_stage(spec, runner, templates, sub_schema_path, generator_path, options),
    }
    metrics = compute_inference_metrics(spec, selector_path, generator_path)
    write_json_atomic(output_directory / "metrics.json", metrics)
    manifest = {
        "method": method,
        "dataset": spec.name,
        "checkpoint": asdict(checkpoint),
        "options": asdict(options),
        "files": {
            "selector_predictions": selector_path.name,
            "predicted_subschemas": sub_schema_path.name,
            "generator_predictions": generator_path.name,
            "metrics": "metrics.json",
        },
        "stages": stages,
    }
    write_json_atomic(output_directory / "manifest.json", manifest)
    return manifest
