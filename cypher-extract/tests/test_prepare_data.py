import json
from pathlib import Path

import pytest

from distillation.prepare_data import LAYOUT_FILE, SPLIT_FILES, convert_directory, to_openai_record


def _prepared_record(task: str = "generator") -> dict[str, str]:
    response = '{"cypher":"MATCH (n) RETURN n"}' if task == "generator" else '{"label": "YES"}'
    return {
        "task": task,
        "system_prompt": "System",
        "user_prompt": "Question",
        "response": response,
        "example_id": f"example:{task}",
    }


def test_to_openai_record_keeps_only_three_supervised_messages() -> None:
    assert to_openai_record(_prepared_record()) == {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": '{"cypher":"MATCH (n) RETURN n"}'},
        ]
    }


def test_convert_directory_writes_all_splits_and_registry(tmp_path: Path) -> None:
    input_dir = tmp_path / "prepared"
    output_dir = tmp_path / "llamafactory"
    input_dir.mkdir()
    for index, filename in enumerate(SPLIT_FILES.values()):
        records = [_prepared_record("generator"), _prepared_record("selector")][: 1 + index % 2]
        (input_dir / filename).write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "batch_size": 2,
                "input_directory": str(tmp_path / "grounding"),
                "seed": 42,
                "preparation_fingerprint": "abc123",
                "files": {"train": {"rows": 1, "selector": 0}},
                "train_selector_contrast_pairs": 0,
                "train_selector_unpaired_negatives": 0,
            }
        ),
        encoding="utf-8",
    )

    counts = convert_directory(input_dir, output_dir)
    assert counts == {"train": 1, "eval": 2, "test_generator": 1, "test_selector": 2}

    registry = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert set(registry) == {f"cypher_prepared_{split}" for split in SPLIT_FILES}
    assert registry["cypher_prepared_train"]["formatting"] == "openai"
    train = json.loads((output_dir / "cypher_prepared_train.jsonl").read_text(encoding="utf-8"))
    assert [message["role"] for message in train["messages"]] == ["system", "user", "assistant"]
    layout = json.loads((output_dir / LAYOUT_FILE).read_text(encoding="utf-8"))
    assert layout["batch_size"] == 2
    assert layout["input_directory"] == str(tmp_path / "grounding")
    assert layout["preparation_seed"] == 42
    assert layout["preparation_fingerprint"] == "abc123"
    assert layout["train_rows"] == 1

    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_directory(input_dir, output_dir)


def test_to_openai_record_rejects_missing_fields_and_unknown_tasks() -> None:
    invalid = _prepared_record()
    invalid["response"] = ""
    with pytest.raises(ValueError, match="response"):
        to_openai_record(invalid)

    with pytest.raises(ValueError, match="Unsupported prepared task"):
        to_openai_record(_prepared_record("ranking"))
