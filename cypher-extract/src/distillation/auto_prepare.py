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

from cypher_extract.paths import get_data_root

from .data_cache import GROUNDING_FILENAMES, preparation_fingerprint
from .prepare_data import LAYOUT_FILE, SPLIT_FILES

MANAGED_TRAIN_DATASET = "cypher_prepared_train"
MANAGED_EVAL_DATASET = "cypher_prepared_eval"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUNDING_INPUT = "cypherbench_schema_grounding_full_final"
DEFAULT_PREPARED_ROOT = "prepared"
# Each managed LlamaFactory root is built from exactly one grounding source, so
# selecting a dataset_dir also selects its source unless an override is given.
# Keys are dataset_dir folder names; sources are sibling folders in the same
# data root, so the pairing holds for both CYPHER_DATA_ROOT and data/ layouts.
MANAGED_DATA_SOURCES = {
    "llamafactory": (DEFAULT_GROUNDING_INPUT, DEFAULT_PREPARED_ROOT),
    "llamafactory_distractor_v1": (
        "cypherbench_schema_grounding_distractor_v1",
        "prepared_distractor_v1",
    ),
}


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
    grounding_input: str | None = None,
    prepared_root: str | None = None,
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
    managed_sources = MANAGED_DATA_SOURCES.get(dataset_root.name)
    if managed_sources is None:
        source_root = get_data_root()
        default_grounding, default_prepared = DEFAULT_GROUNDING_INPUT, DEFAULT_PREPARED_ROOT
    else:
        source_root = dataset_root.parent
        default_grounding, default_prepared = managed_sources
    grounding_input = grounding_input or str(source_root / default_grounding)
    prepared_root = prepared_root or str(source_root / default_prepared)
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
        if int(layout.get("batch_size", -1)) != plan.batch_size:
            return False
    except (TypeError, ValueError):
        return False

    fingerprint_keys = {
        "input_directory",
        "preparation_seed",
        "preparation_fingerprint",
    }
    if fingerprint_keys.isdisjoint(layout):
        # Published dataset snapshots are portable and intentionally omit the
        # machine-specific source path and fingerprint. Their complete file
        # set plus matching batch layout is sufficient for direct reuse.
        return True
    if not fingerprint_keys.issubset(layout):
        return False
    try:
        current_fingerprint = preparation_fingerprint(
            plan.grounding_input_dir,
            plan.prompt_root,
            seed=plan.preparation_seed,
        )
        return (
            Path(str(layout["input_directory"])).resolve()
            == plan.grounding_input_dir.resolve()
            and int(layout["preparation_seed"]) == plan.preparation_seed
            and layout["preparation_fingerprint"] == current_fingerprint
        )
    except (OSError, TypeError, ValueError):
        return False


def _cached_input_directory(plan: AutoPreparePlan) -> Path | None:
    """Return the grounding source recorded by an existing cache, if any."""

    try:
        layout = json.loads((plan.dataset_dir / LAYOUT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(layout, dict) or not isinstance(layout.get("input_directory"), str):
        return None
    return Path(layout["input_directory"]).resolve()


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
    cached_source = _cached_input_directory(plan)
    if not force and cached_source is not None and cached_source != plan.grounding_input_dir.resolve():
        raise RuntimeError(
            f"Refusing to rebuild {plan.dataset_dir}: it was prepared from {cached_source}, not "
            f"{plan.grounding_input_dir.resolve()}. Use a separate dataset_dir per grounding source, "
            "or set AUTO_PREPARE_FORCE=1 to replace it."
        )
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
        # Unset overrides follow the source paired with the config's dataset_dir.
        grounding_input=os.environ.get("CYPHER_GROUNDING_INPUT_DIR") or None,
        prepared_root=os.environ.get("CYPHER_PREPARED_ROOT") or None,
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
