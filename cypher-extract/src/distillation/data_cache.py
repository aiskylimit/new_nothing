"""Stable fingerprints for batch-specific prepared training data."""

from __future__ import annotations

import hashlib
from pathlib import Path

GROUNDING_FILENAMES = tuple(
    f"{task}_{split}.jsonl"
    for task in ("generation", "selection")
    for split in ("train", "dev", "test")
)
PROMPT_FILENAMES = (
    "generator/system_prompt.txt",
    "generator/user_prompt.txt",
    "selector/system_prompt.txt",
    "selector/user_prompt.txt",
)
FINGERPRINT_VERSION = 1


def preparation_fingerprint(
    grounding_input_dir: Path,
    prompt_root: Path,
    *,
    seed: int,
) -> str:
    """Hash every input that can change prepared row content or ordering."""

    digest = hashlib.sha256()
    digest.update(f"version={FINGERPRINT_VERSION}\nseed={seed}\n".encode())
    inputs = [
        *((f"grounding/{name}", grounding_input_dir / name) for name in GROUNDING_FILENAMES),
        *((f"prompts/{name}", prompt_root / name) for name in PROMPT_FILENAMES),
    ]
    for logical_name, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(logical_name.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
