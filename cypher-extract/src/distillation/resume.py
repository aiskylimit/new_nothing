from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_RANKED_RNG_RE = re.compile(r"^rng_state_(\d+)\.pth$")
_DISTILLM_STATE_RE = re.compile(r"^distillm_state_rank(\d+)\.pt$")
_DEEPSPEED_OPTIM_RANK_RE = re.compile(r"zero_pp_rank_(\d+)")
RESUME_MANIFEST_NAME = "resume_manifest.json"
_RESUME_CONFIG_IGNORED_FIELDS = {
    "model_revision",
    "ref_model_revision",
    "resume_from_checkpoint",
    "save_total_limit",
}
_MODEL_STATE_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_MODEL_IDENTITY_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_WEIGHT_SAMPLE_BYTES = 64 * 1024


class ResumeCheckpointError(ValueError):
    """Raised when a checkpoint cannot restore the complete training state."""


def _is_full_model_weight(path: Path) -> bool:
    return (
        path.name == "model.safetensors"
        or (path.name.startswith("model-") and path.suffix == ".safetensors")
        or path.name == "pytorch_model.bin"
        or (path.name.startswith("pytorch_model-") and path.suffix == ".bin")
    )


def _file_sha256(path: Path, *, sampled: bool = False) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if not sampled:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        else:
            digest.update(handle.read(_WEIGHT_SAMPLE_BYTES))
            if path.stat().st_size > _WEIGHT_SAMPLE_BYTES:
                handle.seek(max(path.stat().st_size - _WEIGHT_SAMPLE_BYTES, 0))
                digest.update(handle.read(_WEIGHT_SAMPLE_BYTES))
    return digest.hexdigest()


def full_model_checkpoint_identity(value: str | None) -> dict[str, Any] | None:
    """Fingerprint local full-model assets without hashing every multi-GB shard."""

    if not value:
        return None
    path = Path(value).expanduser().resolve()
    identity: dict[str, Any] = {"source": value}
    if not path.is_dir():
        return identity

    files = []
    for candidate in sorted((item for item in path.iterdir() if item.is_file()), key=lambda item: item.name):
        if _is_full_model_weight(candidate):
            files.append(
                {
                    "name": candidate.name,
                    "size": candidate.stat().st_size,
                    "sample_sha256": _file_sha256(candidate, sampled=True),
                }
            )
        elif candidate.name in _MODEL_IDENTITY_FILES:
            files.append({"name": candidate.name, "sha256": _file_sha256(candidate)})
    identity.update(resolved_path=str(path), files=files)
    return identity


def validate_fresh_output_dir(output_dir: str | os.PathLike[str]) -> None:
    """Reject a new run that could coexist with checkpoints from an older run."""

    directory = Path(output_dir).expanduser().resolve()
    if not directory.is_dir():
        return
    checkpoints = sorted(
        (path for path in directory.iterdir() if path.is_dir() and _CHECKPOINT_RE.fullmatch(path.name)),
        key=lambda path: int(_CHECKPOINT_RE.fullmatch(path.name).group(1)),
    )
    if checkpoints:
        names = ", ".join(path.name for path in checkpoints)
        raise ResumeCheckpointError(
            f"Refusing to start fresh training in {directory}: existing checkpoints: {names}. "
            "Use a new output_dir/RESULTS_ROOT, or pass one explicit resume_from_checkpoint path."
        )


def canonical_resume_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable, JSON-safe config without the checkpoint path itself."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, list | tuple):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, str | int | float | bool):
            return value
        return str(value)

    return {
        str(key): normalize(value)
        for key, value in sorted(config.items(), key=lambda pair: str(pair[0]))
        if key not in _RESUME_CONFIG_IGNORED_FIELDS
    }


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    payload = json.dumps(canonical_resume_config(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_resume_manifest(
    directory: str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
    world_size: int,
    global_step: int,
    runtime_identity: Mapping[str, Any] | None = None,
) -> None:
    """Persist the run identity used to reject incompatible future resumes."""

    normalized_config = canonical_resume_config(config)
    normalized_runtime = canonical_resume_config(runtime_identity or {})
    manifest = {
        "format_version": 2,
        "global_step": int(global_step),
        "world_size": int(world_size),
        "config_sha256": _payload_fingerprint(normalized_config),
        "config": normalized_config,
        "runtime_sha256": _payload_fingerprint(normalized_runtime),
        "runtime": normalized_runtime,
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / RESUME_MANIFEST_NAME
    temporary = directory / f".{RESUME_MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _rank_set(directory: Path, pattern: re.Pattern[str]) -> set[int]:
    ranks: set[int] = set()
    for item in directory.iterdir():
        match = pattern.fullmatch(item.name)
        if item.is_file() and match is not None:
            ranks.add(int(match.group(1)))
    return ranks


def _require_exact_ranks(
    errors: list[str],
    *,
    actual: set[int],
    world_size: int,
    description: str,
) -> None:
    expected = set(range(world_size))
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing ranks {missing}")
    if extra:
        details.append(f"unexpected ranks {extra}")
    errors.append(f"{description}: {', '.join(details)} (current world_size={world_size})")


def _load_trainer_state(checkpoint: Path, errors: list[str]) -> dict[str, Any] | None:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        errors.append("missing trainer_state.json")
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read trainer_state.json: {exc}")
        return None
    if not isinstance(state, dict):
        errors.append("trainer_state.json must contain a JSON object")
        return None
    return state


def _validate_resume_manifest(
    checkpoint: Path,
    *,
    expected_config: Mapping[str, Any] | None,
    expected_runtime: Mapping[str, Any] | None,
    step: int,
    world_size: int,
    errors: list[str],
) -> None:
    if expected_config is None:
        return
    manifest_path = checkpoint / RESUME_MANIFEST_NAME
    if not manifest_path.is_file():
        warnings.warn(
            f"{checkpoint} predates {RESUME_MANIFEST_NAME}; structural state was validated, but the old run config "
            "cannot be fingerprint-checked. Keep the original config unchanged.",
            stacklevel=2,
        )
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {RESUME_MANIFEST_NAME}: {exc}")
        return
    if not isinstance(manifest, dict):
        errors.append(f"{RESUME_MANIFEST_NAME} must contain a JSON object")
        return
    format_version = manifest.get("format_version")
    if format_version not in {1, 2}:
        errors.append(f"unsupported resume manifest format {manifest.get('format_version')!r}")
    if manifest.get("global_step") != step:
        errors.append(f"resume manifest step {manifest.get('global_step')!r} does not match checkpoint step {step}")
    if manifest.get("world_size") != world_size:
        errors.append(
            f"resume manifest world_size={manifest.get('world_size')!r} does not match current world_size={world_size}"
        )

    normalized_expected = canonical_resume_config(expected_config)
    expected_fingerprint = _payload_fingerprint(normalized_expected)
    if manifest.get("config_sha256") != expected_fingerprint:
        saved_config = manifest.get("config")
        changed_keys: list[str] = []
        if isinstance(saved_config, dict):
            changed_keys = sorted(
                key
                for key in set(saved_config) | set(normalized_expected)
                if saved_config.get(key) != normalized_expected.get(key)
            )
        detail = f"; changed fields: {', '.join(changed_keys)}" if changed_keys else ""
        errors.append(f"training config fingerprint does not match the checkpoint{detail}")

    if expected_runtime is None:
        return
    if format_version == 1 or not isinstance(manifest.get("runtime"), dict):
        warnings.warn(
            f"{checkpoint} has no runtime/content fingerprint; config and structural state were validated only.",
            stacklevel=2,
        )
        return
    normalized_runtime = canonical_resume_config(expected_runtime)
    if manifest.get("runtime_sha256") != _payload_fingerprint(normalized_runtime):
        saved_runtime = manifest["runtime"]
        changed_keys = sorted(
            key
            for key in set(saved_runtime) | set(normalized_runtime)
            if saved_runtime.get(key) != normalized_runtime.get(key)
        )
        detail = f"; changed sections: {', '.join(changed_keys)}" if changed_keys else ""
        errors.append(f"runtime/data/model fingerprint does not match the checkpoint{detail}")


def _validate_deepspeed_state(checkpoint: Path, step: int, world_size: int, errors: list[str]) -> None:
    expected_tag = f"global_step{step}"
    latest_path = checkpoint / "latest"
    if not latest_path.is_file():
        errors.append("missing DeepSpeed latest tag file")
        latest_tag = None
    else:
        try:
            latest_tag = latest_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read DeepSpeed latest tag: {exc}")
            latest_tag = None
        if latest_tag and latest_tag != expected_tag:
            errors.append(f"DeepSpeed latest points to {latest_tag!r}, expected {expected_tag!r}")

    state_dir = checkpoint / expected_tag
    if not state_dir.is_dir():
        errors.append(f"missing DeepSpeed state directory {expected_tag}/")
        return

    model_states = list(state_dir.glob("*model_states.pt"))
    if not model_states:
        errors.append(f"{expected_tag}/ has no DeepSpeed model state")

    optimizer_states = list(state_dir.glob("*optim_states.pt"))
    optimizer_ranks = {
        int(match.group(1))
        for path in optimizer_states
        if (match := _DEEPSPEED_OPTIM_RANK_RE.search(path.name)) is not None
    }
    _require_exact_ranks(
        errors,
        actual=optimizer_ranks,
        world_size=world_size,
        description="incomplete DeepSpeed optimizer shards",
    )


def _validate_standard_state(checkpoint: Path, errors: list[str]) -> None:
    if not any((checkpoint / name).is_file() for name in _MODEL_STATE_NAMES):
        errors.append("missing model/adapter weights")
    if not any((checkpoint / name).is_file() for name in ("optimizer.pt", "optimizer.bin")):
        errors.append("missing optimizer state (optimizer.pt or optimizer.bin)")
    if not (checkpoint / "scheduler.pt").is_file():
        errors.append("missing scheduler.pt")


def validate_resume_checkpoint(
    resume_from_checkpoint: str | os.PathLike[str] | bool | None,
    *,
    output_dir: str | os.PathLike[str],
    world_size: int,
    deepspeed_enabled: bool,
    require_distillm_state: bool,
    train_batch_size: int | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_runtime: Mapping[str, Any] | None = None,
) -> Path | None:
    """Resolve and validate an explicit checkpoint before resuming training.

    Exact continuation needs more than model weights. This validates Trainer,
    optimizer/scheduler, RNG, distributed-rank, and adaptive DistiLLM state.
    """

    if resume_from_checkpoint is None or resume_from_checkpoint is False:
        return None
    if resume_from_checkpoint is True:
        raise ResumeCheckpointError(
            "resume_from_checkpoint must be an explicit checkpoint path; "
            "automatic latest-checkpoint resume is disabled."
        )
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    raw_checkpoint = os.fspath(resume_from_checkpoint).strip()
    if raw_checkpoint.lower() in {"auto", "latest", "true"}:
        raise ResumeCheckpointError(
            "resume_from_checkpoint must name checkpoint-N explicitly; automatic latest-checkpoint resume is disabled."
        )

    checkpoint = Path(raw_checkpoint).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    errors: list[str] = []

    if not checkpoint.is_dir():
        raise ResumeCheckpointError(f"Resume checkpoint does not exist or is not a directory: {checkpoint}")

    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        errors.append("checkpoint directory name must be checkpoint-N")
        step = -1
    else:
        step = int(match.group(1))

    if checkpoint.parent != resolved_output_dir:
        errors.append(
            f"checkpoint must be directly inside output_dir ({resolved_output_dir}); got parent {checkpoint.parent}"
        )

    trainer_state = _load_trainer_state(checkpoint, errors)
    if trainer_state is not None and step >= 0:
        saved_step = trainer_state.get("global_step")
        if isinstance(saved_step, bool) or not isinstance(saved_step, int) or saved_step != step:
            errors.append(f"trainer_state global_step={saved_step!r} does not match checkpoint step {step}")
        saved_batch_size = trainer_state.get("train_batch_size")
        if train_batch_size is not None and saved_batch_size is not None and saved_batch_size != train_batch_size:
            errors.append(
                f"train_batch_size changed from {saved_batch_size!r} to {train_batch_size!r}; "
                "resume with the original per-device batch size"
            )

    if step >= 0:
        _validate_resume_manifest(
            checkpoint,
            expected_config=expected_config,
            expected_runtime=expected_runtime,
            step=step,
            world_size=world_size,
            errors=errors,
        )

    ranked_rng = _rank_set(checkpoint, _RANKED_RNG_RE)
    if world_size == 1:
        if not (checkpoint / "rng_state.pth").is_file():
            errors.append("missing single-process RNG state rng_state.pth")
        if ranked_rng:
            errors.append(f"found distributed RNG states for ranks {sorted(ranked_rng)} but current world_size=1")
    else:
        if (checkpoint / "rng_state.pth").is_file():
            errors.append("found single-process RNG state but current run is distributed")
        _require_exact_ranks(
            errors,
            actual=ranked_rng,
            world_size=world_size,
            description="incomplete per-rank RNG states",
        )

    if deepspeed_enabled:
        if step >= 0:
            _validate_deepspeed_state(checkpoint, step, world_size, errors)
    else:
        _validate_standard_state(checkpoint, errors)

    if require_distillm_state:
        _require_exact_ranks(
            errors,
            actual=_rank_set(checkpoint, _DISTILLM_STATE_RE),
            world_size=world_size,
            description="incomplete adaptive DistiLLM states",
        )

    if errors:
        formatted = "\n  - ".join(errors)
        raise ResumeCheckpointError(f"Checkpoint {checkpoint} cannot be resumed exactly:\n  - {formatted}")
    return checkpoint
