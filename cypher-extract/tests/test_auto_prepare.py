from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from distillation.auto_prepare import (
    AutoPreparePlan,
    build_auto_prepare_plan,
    cache_is_ready,
    ensure_training_data,
)
from distillation.data_cache import (
    GROUNDING_FILENAMES,
    PROMPT_FILENAMES,
    preparation_fingerprint,
)
from distillation.prepare_data import LAYOUT_FILE, SPLIT_FILES


def _write_config(path: Path, *, batch_size: int = 2, dataset: str = "cypher_prepared_train") -> None:
    path.write_text(
        "\n".join(
            [
                "do_train: true",
                f"dataset: {dataset}",
                "eval_dataset: cypher_prepared_eval",
                "dataset_dir: data/llamafactory",
                f"per_device_train_batch_size: {batch_size}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_plan_uses_batch_specific_cache_and_honors_cli_override(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    _write_config(config, batch_size=2)

    plan = build_auto_prepare_plan(
        config,
        ["per_device_train_batch_size=8"],
        project_root=tmp_path,
    )

    assert plan is not None
    assert plan.batch_size == 8
    assert plan.prepared_dir == tmp_path / "data" / "prepared" / "batch_8"
    assert plan.dataset_dir == tmp_path / "data" / "llamafactory" / "batch_8"
    assert plan.dataset_dir_override == "data/llamafactory/batch_8"


def test_plan_skips_non_managed_dataset(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    _write_config(config, dataset="external_train")

    assert build_auto_prepare_plan(config, project_root=tmp_path) is None


def _write_cache_inputs(plan: AutoPreparePlan) -> None:
    plan.grounding_input_dir.mkdir(parents=True, exist_ok=True)
    for filename in GROUNDING_FILENAMES:
        path = plan.grounding_input_dir / filename
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    for filename in PROMPT_FILENAMES:
        path = plan.prompt_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"prompt:{filename}\n", encoding="utf-8")


def _publish_ready_cache(plan: AutoPreparePlan) -> None:
    _write_cache_inputs(plan)
    plan.dataset_dir.mkdir(parents=True, exist_ok=True)
    (plan.dataset_dir / "dataset_info.json").write_text("{}\n", encoding="utf-8")
    (plan.dataset_dir / LAYOUT_FILE).write_text(
        json.dumps(
            {
                "batch_size": plan.batch_size,
                "input_directory": str(plan.grounding_input_dir.resolve()),
                "preparation_seed": plan.preparation_seed,
                "preparation_fingerprint": preparation_fingerprint(
                    plan.grounding_input_dir,
                    plan.prompt_root,
                    seed=plan.preparation_seed,
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for split in SPLIT_FILES:
        (plan.dataset_dir / f"cypher_prepared_{split}.jsonl").write_text("{}\n", encoding="utf-8")


def test_cache_readiness_requires_matching_layout_and_all_files(tmp_path: Path) -> None:
    plan = AutoPreparePlan(
        batch_size=4,
        grounding_input_dir=tmp_path / "grounding",
        prepared_dir=tmp_path / "prepared",
        dataset_dir=tmp_path / "llamafactory",
        dataset_dir_override="llamafactory",
        prompt_root=tmp_path / "prompts",
    )
    assert not cache_is_ready(plan)

    _publish_ready_cache(plan)
    assert cache_is_ready(plan)

    (plan.dataset_dir / LAYOUT_FILE).write_text('{"batch_size": 8}\n', encoding="utf-8")
    assert not cache_is_ready(plan)

    (plan.dataset_dir / LAYOUT_FILE).write_text("[]\n", encoding="utf-8")
    assert not cache_is_ready(plan)

    (plan.dataset_dir / LAYOUT_FILE).write_text('{"batch_size": null}\n', encoding="utf-8")
    assert not cache_is_ready(plan)


def test_cache_is_invalidated_by_source_content_directory_prompt_and_seed(tmp_path: Path) -> None:
    plan = AutoPreparePlan(
        batch_size=4,
        grounding_input_dir=tmp_path / "grounding-a",
        prepared_dir=tmp_path / "prepared",
        dataset_dir=tmp_path / "llamafactory",
        dataset_dir_override="llamafactory",
        prompt_root=tmp_path / "prompts",
    )
    _publish_ready_cache(plan)
    assert cache_is_ready(plan)

    changed_source = plan.grounding_input_dir / GROUNDING_FILENAMES[0]
    changed_source.write_text('{"changed": true}\n', encoding="utf-8")
    assert not cache_is_ready(plan)
    _publish_ready_cache(plan)

    changed_prompt = plan.prompt_root / PROMPT_FILENAMES[0]
    changed_prompt.write_text("changed prompt\n", encoding="utf-8")
    assert not cache_is_ready(plan)

    other_directory = replace(plan, grounding_input_dir=tmp_path / "grounding-b")
    _write_cache_inputs(other_directory)
    assert not cache_is_ready(other_directory)
    assert not cache_is_ready(replace(plan, preparation_seed=99))


def test_ensure_builds_missing_cache_once(tmp_path: Path) -> None:
    plan = AutoPreparePlan(
        batch_size=4,
        grounding_input_dir=tmp_path / "grounding",
        prepared_dir=tmp_path / "prepared",
        dataset_dir=tmp_path / "llamafactory",
        dataset_dir_override="llamafactory",
        prompt_root=tmp_path / "prompts",
    )
    _write_cache_inputs(plan)

    commands: list[list[str]] = []

    def fake_run(command: list[str], _project_root: Path) -> None:
        commands.append(command)
        if "prepare_llamafactory_data.py" in command[1]:
            _publish_ready_cache(plan)

    assert ensure_training_data(plan, project_root=tmp_path, run_command=fake_run)
    assert [Path(command[1]).name for command in commands] == [
        "prepare_multitask_prompts.py",
        "prepare_llamafactory_data.py",
    ]
    assert not ensure_training_data(plan, project_root=tmp_path, run_command=fake_run)
    assert len(commands) == 2
