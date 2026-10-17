from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_hf_path(path: str) -> tuple[str, str]:
    if not path.startswith("hf://"):
        raise ValueError(f"Not a Hugging Face URI: {path!r}.")
    parts = path.removeprefix("hf://").strip("/").split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError("Expected hf://<owner>/<repo>/<optional/subdirectory>.")
    return "/".join(parts[:2]), "/".join(parts[2:])


@cache
def resolve_hf_path(path: str | None, *, repo_type: str | None = None) -> str | None:
    """Resolve local paths and `hf://repo/subdirectory` URIs.

    `repo_type="auto"` tries a model repository first and then a dataset
    repository, matching the behavior of the template dataset loader.
    """

    if path is None:
        return None

    if not path.startswith("hf://"):
        return path

    repo_id, subdirectory = parse_hf_path(path)
    allow_patterns = [f"{subdirectory}/*", f"{subdirectory}/**"] if subdirectory else None
    token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    repo_types: tuple[str | None, ...]
    if repo_type == "auto":
        repo_types = (None, "dataset")
    else:
        repo_types = (repo_type,)

    last_error: Exception | None = None
    for current_repo_type in repo_types:
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                repo_type=current_repo_type,
                allow_patterns=allow_patterns,
                token=token,
            )
            resolved = Path(snapshot_path, subdirectory) if subdirectory else Path(snapshot_path)
            if not resolved.is_dir():
                raise FileNotFoundError(f"Resolved Hugging Face path does not exist: {resolved}")
            return str(resolved)
        except Exception as exc:  # retrying the alternate repo type is intentional
            last_error = exc

    raise RuntimeError(f"Failed to resolve Hugging Face path {path!r}.") from last_error
