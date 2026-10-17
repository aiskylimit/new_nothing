from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT_ROOT = Path("results/lora")
DEFAULT_MODEL_FAMILY = "qwen3"
SUPPORTED_MODEL_FAMILIES = ("qwen3", "llama3", "qwen2.5_coder")
DEFAULT_METHODS = (
    "teacher_lora",
    "sft",
    "fkl",
    "rkl",
    "sfkl",
    "srkl",
    "csd",
    "hpd",
    "amid",
    "fdd_sfkl",
    "fdd_srkl",
    "distillm_adaptive_sfkl",
    "distillm_adaptive_srkl",
)
SUPPORTED_METHODS = (*DEFAULT_METHODS, "teacher_full")
_CHECKPOINT_RE = re.compile(r"(?:^|/)checkpoint-(?P<step>\d+)$")
_FULL_HASH_FILES = frozenset(
    {
        "adapter_config.json",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "resume_manifest.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "trainer_state.json",
        "vocab.json",
    }
)
_WEIGHT_SAMPLE_BYTES = 64 * 1024
_RESUME_MANIFEST_NAME = "resume_manifest.json"


@dataclass(frozen=True)
class LastCheckpoint:
    checkpoint_root: str
    model_family: str
    method: str
    step: int
    subfolder: str
    fingerprint: str = ""

    @property
    def path(self) -> Path:
        return Path(self.checkpoint_root, *self.subfolder.split("/"))

    @property
    def uri(self) -> str:
        return str(self.path)


def select_last_checkpoint(paths: Iterable[str], method_prefix: str) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = []
    normalized_prefix = method_prefix.strip("/")
    for path in paths:
        normalized_path = path.replace("\\", "/").rstrip("/")
        match = _CHECKPOINT_RE.search(normalized_path)
        if match and normalized_path.startswith(f"{normalized_prefix}/"):
            candidates.append((int(match.group("step")), normalized_path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-N directory found under {normalized_prefix}")
    return max(candidates, key=lambda item: item[0])


def _is_weight_file(path: Path) -> bool:
    return path.name in {"adapter_model.bin", "adapter_model.safetensors"} or (
        path.name.startswith("model-") and path.suffix == ".safetensors"
    ) or path.name == "model.safetensors" or (
        path.name.startswith("pytorch_model-") and path.suffix == ".bin"
    ) or path.name == "pytorch_model.bin"


def _file_sha256(path: Path, *, sampled: bool) -> str:
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


def checkpoint_fingerprint(directory: Path) -> str:
    """Fingerprint inference assets without hashing complete multi-GB model shards."""

    entries = []
    for path in sorted((item for item in directory.iterdir() if item.is_file()), key=lambda item: item.name):
        if _is_weight_file(path):
            stat = path.stat()
            entries.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sample_sha256": _file_sha256(path, sampled=True),
                }
            )
        elif path.name in _FULL_HASH_FILES:
            entries.append({"name": path.name, "sha256": _file_sha256(path, sampled=False)})
    serialized = json.dumps(entries, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(f"checkpoint-fingerprint-v1\0{serialized}".encode()).hexdigest()


def _run_signature(directory: Path, step: int) -> tuple[int, str, str | None] | None:
    manifest_path = directory / _RESUME_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    if manifest.get("global_step") != step:
        raise ValueError(
            f"{manifest_path} global_step={manifest.get('global_step')!r} does not match checkpoint-{step}"
        )
    format_version = manifest.get("format_version")
    config_sha256 = manifest.get("config_sha256")
    if format_version not in {1, 2}:
        raise ValueError(f"{manifest_path} has unsupported format_version={format_version!r}")
    if not isinstance(config_sha256, str) or not config_sha256:
        raise ValueError(f"{manifest_path} has no valid config_sha256")
    runtime_sha256 = manifest.get("runtime_sha256")
    if format_version >= 2 and (not isinstance(runtime_sha256, str) or not runtime_sha256):
        raise ValueError(f"{manifest_path} has no valid runtime_sha256")
    return format_version, config_sha256, runtime_sha256


def resolve_last_checkpoint(
    method: str,
    *,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> LastCheckpoint:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected one of {', '.join(SUPPORTED_METHODS)}")
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported model family {model_family!r}; expected one of {', '.join(SUPPORTED_MODEL_FAMILIES)}"
        )

    root = Path(checkpoint_root).expanduser().resolve()
    method_prefix = f"{model_family}/{method}"
    method_directory = root / model_family / method
    if not method_directory.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {method_directory}")

    checkpoint_directories = []
    signed_candidates = []
    for path in method_directory.iterdir():
        if not path.is_dir():
            continue
        match = _CHECKPOINT_RE.search(path.relative_to(root).as_posix())
        if match is None:
            continue
        step = int(match.group("step"))
        checkpoint_directories.append(path)
        signature = _run_signature(path, step)
        if signature is not None:
            signed_candidates.append((path, signature))

    if signed_candidates:
        newest_format = max(signature[0] for _, signature in signed_candidates)
        signed_candidates = [item for item in signed_candidates if item[1][0] == newest_format]
        run_signatures = {signature[1:] for _, signature in signed_candidates}
        if len(run_signatures) != 1:
            raise ValueError(
                f"Mixed training-run fingerprints found under {method_directory}; "
                "use a clean checkpoint root for one run."
            )
        checkpoint_directories = [path for path, _ in signed_candidates]

    paths = (path.relative_to(root).as_posix() for path in checkpoint_directories)
    step, subfolder = select_last_checkpoint(paths, method_prefix)
    directory = root / Path(*subfolder.split("/"))
    return LastCheckpoint(
        str(root),
        model_family,
        method,
        step,
        subfolder,
        fingerprint=checkpoint_fingerprint(directory),
    )


def resolve_checkpoint_directory(checkpoint: LastCheckpoint) -> Path:
    """Validate and return a local adapter or full-model checkpoint directory."""

    directory = checkpoint.path
    if not directory.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {directory}")

    adapter_config = directory / "adapter_config.json"
    if adapter_config.is_file():
        if not (directory / "adapter_model.safetensors").is_file() and not (
            directory / "adapter_model.bin"
        ).is_file():
            raise FileNotFoundError(f"Missing adapter weights in {directory}")
    else:
        full_model_files = list(directory.glob("model*.safetensors")) + list(
            directory.glob("pytorch_model*.bin")
        )
        if not (directory / "config.json").is_file() or not full_model_files:
            raise FileNotFoundError(f"Missing adapter or full-model weights in {directory}")
    return directory
