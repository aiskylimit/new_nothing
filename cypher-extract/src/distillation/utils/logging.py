from __future__ import annotations

from pathlib import Path
from typing import Any

from .distributed import get_rank


def print_rank(*values: Any, rank: int = 0, **kwargs: Any) -> None:
    if get_rank() == rank:
        print(*values, **kwargs)


def save_rank(message: str, path: str | Path, *, rank: int = 0) -> None:
    if get_rank() != rank:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
