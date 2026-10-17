#!/usr/bin/env python3
"""Measure token lengths that ``cutoff_len`` and inference context limits must cover.

Two questions are answered per model family:

``training``
    How long is the longest rendered ``prompt + response`` sequence in the
    prepared LlamaFactory training data? ``cutoff_len`` truncates exactly this,
    so it must be at least the reported maximum.

``inference``
    How long is the longest prompt the two-stage pipeline can actually build?
    The generator is prompted with the *predicted* sub-schema, so the worst
    case is every schema unit of an example being selected, not the gold
    sub-schema. Prompts are built with the same ``PromptTemplates`` the
    pipeline uses, so the numbers keep prompt parity by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cypher_extract.paths import get_data_root  # noqa: E402
from schema_grounding.inference.data import default_dataset_specs  # noqa: E402
from schema_grounding.inference.merge import merge_schema_units  # noqa: E402
from schema_grounding.inference.prompting import (  # noqa: E402
    PromptTemplates,
    render_llama3,
    render_qwen3_nothink,
)

# Tokenizer used to represent each family. Within a family the student and
# teacher share a vocabulary, so one tokenizer characterises the whole family.
MODEL_FAMILIES = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen2.5_coder": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "llama3": "meta-llama/Llama-3.2-1B-Instruct",
}
TRAINING_FILES = {
    "train": "cypher_prepared_train.jsonl",
    "eval": "cypher_prepared_eval.jsonl",
}
SELECTOR_MAX_NEW_TOKENS = 16
GENERATOR_MAX_NEW_TOKENS = 256
BATCH_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data_root = get_data_root()
    parser.add_argument(
        "--model-families",
        default="all",
        help=f"Comma-separated families or 'all'. Choices: {', '.join(MODEL_FAMILIES)}.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override the tokenizer path. Only valid with a single --model-families value.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=data_root / "llamafactory",
        help="Directory holding the prepared LlamaFactory training files.",
    )
    parser.add_argument("--output", type=Path, default=data_root / "token_length_analysis.json")
    parser.add_argument("--max-rows", type=int, default=None, help="Row limit per file for a smoke test.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download; resolve tokenizers from the local Hugging Face cache only.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        emitted = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_rows is not None and emitted >= max_rows:
                return
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            emitted += 1
            yield row


def make_renderer(family: str, tokenizer: Any):
    """Return the exact renderer LlamaFactory uses for this family."""

    if family == "llama3":
        bos_token = tokenizer.bos_token or ""

        def render(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
            return render_llama3(messages, bos_token=bos_token, add_generation_prompt=add_generation_prompt)

        return render

    def render(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
        return render_qwen3_nothink(messages, add_generation_prompt=add_generation_prompt)

    return render


def _lengths(tokenizer: Any, rendered: list[str]) -> list[int]:
    encoded = tokenizer(rendered, add_special_tokens=False, padding=False, truncation=False, return_length=True)
    return [int(length) for length in encoded["length"]]


def _collect(tokenizer: Any, entries: Iterator[tuple[str, str | None]]) -> dict[str, int]:
    """Stream rendered prompts through the tokenizer, keeping only the maxima."""

    rows = max_prompt = max_total = max_response = 0

    def consume(batch: list[tuple[str, str | None]]) -> None:
        nonlocal rows, max_prompt, max_total, max_response
        prompt_lengths = _lengths(tokenizer, [prompt for prompt, _ in batch])
        measured_full = [full for _, full in batch if full is not None]
        full_lengths = _lengths(tokenizer, measured_full) if measured_full else []
        for index, prompt_length in enumerate(prompt_lengths):
            rows += 1
            max_prompt = max(max_prompt, prompt_length)
            if full_lengths:
                max_total = max(max_total, full_lengths[index])
                max_response = max(max_response, full_lengths[index] - prompt_length)

    batch: list[tuple[str, str | None]] = []
    for entry in entries:
        batch.append(entry)
        if len(batch) >= BATCH_SIZE:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    if rows == 0:
        raise ValueError("No rows were measured")

    report = {"rows": rows, "max_prompt_tokens": max_prompt}
    if max_total:
        report["max_total_tokens"] = max_total
        report["max_response_tokens"] = max_response
    return report


def analyze_training(
    tokenizer: Any,
    render,
    prepared_dir: Path,
    max_rows: int | None,
) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, int]] = {}
    for split, filename in TRAINING_FILES.items():
        path = prepared_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared training data: {path}")

        def entries(path: Path = path) -> Iterator[tuple[str, str | None]]:
            for row in iter_jsonl(path, max_rows):
                messages = row["messages"]
                yield (
                    render(messages[:-1], add_generation_prompt=True),
                    render(messages, add_generation_prompt=False),
                )

        results[split] = _collect(tokenizer, entries())
        values = results[split]
        print(
            f"  training/{split:5} rows={values['rows']:7} "
            f"max_prompt={values['max_prompt_tokens']:5} max_total={values['max_total_tokens']:5}",
            flush=True,
        )
    return results


def _grouped_units(path: Path, max_rows: int | None) -> Iterator[tuple[str, str, list[dict[str, Any]]]]:
    """Yield (example_id, question, units) for each consecutive example run."""

    current_id: str | None = None
    question = ""
    units: list[dict[str, Any]] = []
    for row in iter_jsonl(path, max_rows):
        example_id = str(row["example_id"])
        if example_id != current_id:
            if current_id is not None:
                yield current_id, question, units
            current_id = example_id
            question = str(row["question"])
            units = []
        units.append(row["unit"])
    if current_id is not None:
        yield current_id, question, units


def analyze_inference(
    tokenizer: Any,
    render,
    templates: PromptTemplates,
    max_rows: int | None,
) -> dict[str, dict[str, dict[str, int]]]:
    results: dict[str, dict[str, dict[str, int]]] = {}
    for name, spec in default_dataset_specs(ROOT).items():
        for path in (spec.selection_test, spec.generation_test):
            if not path.is_file():
                raise FileNotFoundError(f"Missing inference data: {path}")

        def selector_entries(spec=spec) -> Iterator[tuple[str, str | None]]:
            for row in iter_jsonl(spec.selection_test, max_rows):
                messages = templates.selector_messages(str(row["question"]), str(row["unit"]["text"]))
                yield render(messages, add_generation_prompt=True), None

        def generator_entries(spec=spec) -> Iterator[tuple[str, str | None]]:
            # Worst case: the selector marks every unit of the example as
            # related, so the predicted sub-schema is the full schema.
            for _, question, units in _grouped_units(spec.selection_test, max_rows):
                merged = merge_schema_units(units, [str(unit["id"]) for unit in units])
                messages = templates.generator_messages(question, merged.sub_schema)
                yield render(messages, add_generation_prompt=True), None

        def gold_entries(spec=spec) -> Iterator[tuple[str, str | None]]:
            for row in iter_jsonl(spec.generation_test, max_rows):
                if "sub_schema" not in row:
                    continue
                messages = templates.generator_messages(str(row["question"]), row["sub_schema"])
                yield render(messages, add_generation_prompt=True), None

        dataset: dict[str, dict[str, int]] = {
            "selector": _collect(tokenizer, selector_entries()),
            "generator_worst_case": _collect(tokenizer, generator_entries()),
        }
        try:
            dataset["generator_gold_subschema"] = _collect(tokenizer, gold_entries())
        except ValueError:
            pass
        dataset["selector"]["required_context"] = (
            dataset["selector"]["max_prompt_tokens"] + SELECTOR_MAX_NEW_TOKENS
        )
        dataset["generator_worst_case"]["required_context"] = (
            dataset["generator_worst_case"]["max_prompt_tokens"] + GENERATOR_MAX_NEW_TOKENS
        )
        results[name] = dataset
        print(
            f"  inference/{name:18} selector_prompt={dataset['selector']['max_prompt_tokens']:5} "
            f"generator_prompt_worst_case={dataset['generator_worst_case']['max_prompt_tokens']:5}",
            flush=True,
        )
    return results


def summarize(family_report: dict[str, Any]) -> dict[str, int]:
    training = family_report["training"]
    inference = family_report["inference"]
    required_cutoff = max(split["max_total_tokens"] for split in training.values())
    required_context = max(
        max(task["required_context"] for task in dataset.values() if "required_context" in task)
        for dataset in inference.values()
    )
    return {
        "required_cutoff_len": required_cutoff,
        "required_inference_context": required_context,
    }


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    families = list(MODEL_FAMILIES) if args.model_families == "all" else [
        item.strip() for item in args.model_families.split(",") if item.strip()
    ]
    unknown = sorted(set(families).difference(MODEL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown model families: {', '.join(unknown)}")
    if args.tokenizer is not None and len(families) != 1:
        raise ValueError("--tokenizer requires exactly one --model-families value")

    templates = PromptTemplates.from_repository(ROOT)
    report: dict[str, Any] = {"model_families": {}}
    for family in families:
        tokenizer_id = args.tokenizer or MODEL_FAMILIES[family]
        print(f"[{family}] tokenizer={tokenizer_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            token=os.getenv("HF_TOKEN") or os.getenv("HF_READ_TOKEN"),
            use_fast=True,
            local_files_only=args.local_files_only,
        )
        tokenizer.model_max_length = 10**9
        render = make_renderer(family, tokenizer)
        family_report: dict[str, Any] = {
            "tokenizer": tokenizer_id,
            "training": analyze_training(tokenizer, render, args.prepared_dir, args.max_rows),
            "inference": analyze_inference(tokenizer, render, templates, args.max_rows),
        }
        family_report["summary"] = summarize(family_report)
        report["model_families"][family] = family_report
        print(f"  summary {family_report['summary']}", flush=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
