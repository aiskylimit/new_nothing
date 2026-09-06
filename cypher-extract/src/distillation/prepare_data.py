"""Convert prepared multitask JSONL into LlamaFactory OpenAI chat data.

The prepared corpus separates the three supervised turns into
``system_prompt``, ``user_prompt`` and ``response``. LlamaFactory's ``openai``
formatter consumes those losslessly as system/user/assistant messages; with
``train_on_prompt: false`` it computes loss only on the assistant response.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

SPLIT_FILES = {
    "train": "train.jsonl",
    "eval": "eval.jsonl",
    "test_generator": "test_generator.jsonl",
    "test_selector": "test_selector.jsonl",
}

DATASET_PREFIX = "cypher_prepared"
LAYOUT_FILE = f"{DATASET_PREFIX}_layout.json"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            yield record


def to_openai_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one prepared generator/selector row into three chat turns."""

    required = ("system_prompt", "user_prompt", "response", "task", "example_id")
    missing = [key for key in required if not isinstance(record.get(key), str) or not record[key].strip()]
    if missing:
        raise ValueError(f"Prepared record has missing or empty fields: {', '.join(missing)}")
    task = record["task"]
    if task not in frozenset({"generator", "selector"}):
        raise ValueError(f"Unsupported prepared task: {task!r}")

    return {
        "messages": [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    }


def convert_file(source: Path, destination: Path) -> int:
    """Convert one JSONL file and return the number of written records."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in _read_jsonl(source):
            handle.write(json.dumps(to_openai_record(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def dataset_info_entry(filename: str) -> dict[str, Any]:
    """Return a LlamaFactory dataset registry entry for an OpenAI JSONL file."""

    return {
        "file_name": filename,
        "formatting": "openai",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function",
            "system_tag": "system",
        },
    }


def convert_directory(input_dir: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, int]:
    """Convert all prepared splits and atomically publish the managed files."""

    sources = {split: input_dir / filename for split, filename in SPLIT_FILES.items()}
    missing_sources = [path for path in sources.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing prepared input files: {', '.join(map(str, missing_sources))}")

    output_names = {split: f"{DATASET_PREFIX}_{split}.jsonl" for split in SPLIT_FILES}
    managed_targets = [output_dir / name for name in output_names.values()]
    managed_targets.append(output_dir / "dataset_info.json")
    input_manifest = input_dir / "manifest.json"
    if input_manifest.is_file():
        managed_targets.append(output_dir / LAYOUT_FILE)
    existing_targets = [path for path in managed_targets if path.exists()]
    if existing_targets and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite LlamaFactory data: "
            f"{', '.join(map(str, existing_targets))}. Pass --overwrite to replace it."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging_dir = Path(temporary)
        counts: dict[str, int] = {}
        registry: dict[str, dict[str, Any]] = {}
        for split, source in sources.items():
            output_name = output_names[split]
            counts[split] = convert_file(source, staging_dir / output_name)
            registry[f"{DATASET_PREFIX}_{split}"] = dataset_info_entry(output_name)

        (staging_dir / "dataset_info.json").write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if input_manifest.is_file():
            manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
            cache_metadata = {}
            if all(
                key in manifest
                for key in ("input_directory", "seed", "preparation_fingerprint")
            ):
                cache_metadata = {
                    "input_directory": str(manifest["input_directory"]),
                    "preparation_seed": int(manifest["seed"]),
                    "preparation_fingerprint": str(manifest["preparation_fingerprint"]),
                }
            layout = {
                "batch_size": int(manifest["batch_size"]),
                **cache_metadata,
                "train_rows": int(manifest["files"]["train"]["rows"]),
                "train_selector_rows": int(manifest["files"]["train"]["selector"]),
                "train_selector_contrast_pairs": int(manifest["train_selector_contrast_pairs"]),
                "train_selector_unpaired_negatives": int(
                    manifest["train_selector_unpaired_negatives"]
                ),
            }
            (staging_dir / LAYOUT_FILE).write_text(
                json.dumps(layout, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        for target in managed_targets:
            (staging_dir / target.name).replace(target)

    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/llamafactory"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    counts = convert_directory(args.input_dir, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
