"""Prepare batch-size-specific local training data before launching torchrun."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .data_cache import GROUNDING_FILENAMES, preparation_fingerprint
from .prepare_data import LAYOUT_FILE, SPLIT_FILES

MANAGED_TRAIN_DATASET = "cypher_prepared_train"
MANAGED_EVAL_DATASET = "cypher_prepared_eval"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AutoPreparePlan:
    batch_size: int
    grounding_input_dir: Path
    prepared_dir: Path
    dataset_dir: Path
    dataset_dir_override: str
    prompt_root: Path = PROJECT_ROOT / "prompts"
    preparation_seed: int = 42


def _load_merged_config(config_path: Path, overrides: Sequence[str]) -> dict[str, Any]:
    config = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_cli(list(overrides))),
        resolve=True,
    )
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping.")
    return config


def _resolve_local_path(value: str, project_root: Path) -> tuple[Path, str]:
    path = Path(value)
    if path.is_absolute():
        return path.resolve(), str(path)
    return (project_root / path).resolve(), path.as_posix()


def build_auto_prepare_plan(
    config_path: Path,
    overrides: Sequence[str] = (),
    *,
    project_root: Path = PROJECT_ROOT,
    grounding_input: str = "data/cypherbench_schema_grounding_full_final",
    prepared_root: str = "data/prepared",
    preparation_seed: int = 42,
) -> AutoPreparePlan | None:
    """Return a cache plan for the managed Cypher datasets, or ``None``."""

    config = _load_merged_config(config_path, overrides)
    if not bool(config.get("do_train", False)):
        return None
    if config.get("dataset") != MANAGED_TRAIN_DATASET:
        return None
    if config.get("eval_dataset") != MANAGED_EVAL_DATASET:
        return None

    batch_size = int(config.get("per_device_train_batch_size", 0))
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Managed multitask data requires a positive even per-device train batch size.")

    raw_dataset_dir = config.get("dataset_dir")
    if not isinstance(raw_dataset_dir, str) or not raw_dataset_dir:
        raise ValueError("Managed multitask data requires a local dataset_dir.")
    dataset_root, dataset_root_override = _resolve_local_path(raw_dataset_dir, project_root)
    grounding_dir, _ = _resolve_local_path(grounding_input, project_root)
    prepared_base, prepared_base_override = _resolve_local_path(prepared_root, project_root)

    if batch_size == 2:
        prepared_dir = prepared_base
        dataset_dir = dataset_root
        dataset_dir_override = dataset_root_override
    else:
        cache_name = f"batch_{batch_size}"
        prepared_dir = prepared_base / cache_name
        dataset_dir = dataset_root / cache_name
        dataset_dir_override = str(Path(dataset_root_override) / cache_name).replace("\\", "/")
    return AutoPreparePlan(
        batch_size=batch_size,
        grounding_input_dir=grounding_dir,
        prepared_dir=prepared_dir,
        dataset_dir=dataset_dir,
        dataset_dir_override=dataset_dir_override,
        prompt_root=project_root / "prompts",
        preparation_seed=preparation_seed,
    )


def cache_is_ready(plan: AutoPreparePlan) -> bool:
    required = [
        plan.dataset_dir / "dataset_info.json",
        plan.dataset_dir / LAYOUT_FILE,
        *(
            plan.dataset_dir / f"cypher_prepared_{split}.jsonl"
            for split in SPLIT_FILES
        ),
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        layout = json.loads((plan.dataset_dir / LAYOUT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(layout, dict):
        return False
    try:
        current_fingerprint = preparation_fingerprint(
            plan.grounding_input_dir,
            plan.prompt_root,
            seed=plan.preparation_seed,
        )
        return (
            int(layout.get("batch_size", -1)) == plan.batch_size
            and Path(str(layout.get("input_directory", ""))).resolve()
            == plan.grounding_input_dir.resolve()
            and int(layout.get("preparation_seed", -1)) == plan.preparation_seed
            and layout.get("preparation_fingerprint") == current_fingerprint
        )
    except (OSError, TypeError, ValueError):
        return False


def _run_command(command: list[str], project_root: Path) -> None:
    subprocess.run(
        command,
        cwd=project_root,
        check=True,
        stdout=sys.stderr,
    )


def ensure_training_data(
    plan: AutoPreparePlan,
    *,
    project_root: Path = PROJECT_ROOT,
    force: bool = False,
    run_command: Callable[[list[str], Path], None] = _run_command,
) -> bool:
    """Build the requested cache when absent; return whether a rebuild ran."""

    if not force and cache_is_ready(plan):
        return False
    required_grounding = [plan.grounding_input_dir / name for name in GROUNDING_FILENAMES]
    missing = [path for path in required_grounding if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot auto-prepare training data; missing grounding files: "
            + ", ".join(str(path) for path in missing)
        )

    print(
        f"Preparing managed training data for per-device batch size {plan.batch_size}...",
        file=sys.stderr,
    )
    run_command(
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_multitask_prompts.py"),
            "--input-dir",
            str(plan.grounding_input_dir),
            "--output-dir",
            str(plan.prepared_dir),
            "--batch-size",
            str(plan.batch_size),
            "--seed",
            str(plan.preparation_seed),
            "--overwrite",
        ],
        project_root,
    )
    run_command(
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_llamafactory_data.py"),
            "--input-dir",
            str(plan.prepared_dir),
            "--output-dir",
            str(plan.dataset_dir),
            "--overwrite",
        ],
        project_root,
    )
    if not cache_is_ready(plan):
        raise RuntimeError(f"Auto-prepared dataset cache is incomplete: {plan.dataset_dir}")
    return True


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise ValueError("Usage: python -m distillation.auto_prepare <config.yaml> [key=value ...]")
    config_path = Path(arguments[0]).resolve()
    project_root = Path(os.environ.get("CYPHER_PROJECT_ROOT", PROJECT_ROOT)).resolve()
    plan = build_auto_prepare_plan(
        config_path,
        arguments[1:],
        project_root=project_root,
        grounding_input=os.environ.get(
            "CYPHER_GROUNDING_INPUT_DIR",
            "data/cypherbench_schema_grounding_full_final",
        ),
        prepared_root=os.environ.get("CYPHER_PREPARED_ROOT", "data/prepared"),
        preparation_seed=int(os.environ.get("CYPHER_PREPARE_SEED", "42")),
    )
    if plan is None:
        return
    ensure_training_data(
        plan,
        project_root=project_root,
        force=os.environ.get("AUTO_PREPARE_FORCE", "0") == "1",
    )
    print(plan.dataset_dir_override)


if __name__ == "__main__":
    main()
