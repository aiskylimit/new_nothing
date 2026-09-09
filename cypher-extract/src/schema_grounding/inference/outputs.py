from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TextIO

import torch

from distillation.utils import capture_rng_state, restore_rng_state


class ResumableJsonl:
    """Append to a partial JSONL and atomically publish it on completion."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.partial_path = path.with_suffix(path.suffix + ".partial")
        self.rng_state_path = path.with_suffix(path.suffix + ".rng_state.pt")
        self.rng_temporary_path = self.rng_state_path.with_suffix(self.rng_state_path.suffix + ".tmp")

    @property
    def complete(self) -> bool:
        return self.path.is_file()

    def progress(self, key: str = "id") -> tuple[int, str | None]:
        if not self.partial_path.exists():
            return 0, None
        count = 0
        last_key: str | None = None
        with self.partial_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Corrupt partial output {self.partial_path}:{line_number}; remove it to restart this stage"
                    ) from error
                value = row.get(key)
                if not isinstance(value, str):
                    raise ValueError(f"Partial row {line_number} has no string {key!r}")
                count += 1
                last_key = value
        return count, last_key

    def open_append(self) -> TextIO:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.partial_path.open("a", encoding="utf-8", newline="\n")

    def restore_rng_progress(self, key: str = "id") -> tuple[int, str | None]:
        """Restore RNG at the last committed batch and discard an uncommitted tail."""

        self.rng_temporary_path.unlink(missing_ok=True)
        partial_lines: list[str] = []
        if self.partial_path.is_file():
            with self.partial_path.open("r", encoding="utf-8") as handle:
                partial_lines = [line for line in handle if line.strip()]

        if not partial_lines:
            if self.rng_state_path.exists():
                raise ValueError(f"Orphaned RNG state without partial output: {self.rng_state_path}")
            return 0, None
        if not self.rng_state_path.is_file():
            # No batch was committed by the RNG-aware writer. Discard the
            # legacy/torn partial and regenerate from the already-reset seed.
            self.partial_path.unlink()
            return 0, None

        saved = torch.load(self.rng_state_path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict):
            raise ValueError(f"Invalid RNG progress state: {self.rng_state_path}")
        completed_rows = saved.get("completed_rows")
        last_key = saved.get("last_key")
        if not isinstance(completed_rows, int) or completed_rows <= 0:
            raise ValueError(f"Invalid completed_rows in {self.rng_state_path}")
        if completed_rows > len(partial_lines):
            raise ValueError(
                f"RNG state records {completed_rows} rows but {self.partial_path} contains only {len(partial_lines)}"
            )
        committed_rows = []
        for line_number, line in enumerate(partial_lines[:completed_rows], 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Corrupt committed output {self.partial_path}:{line_number}") from error
            value = row.get(key)
            if not isinstance(value, str):
                raise ValueError(f"Committed partial row {line_number} has no string {key!r}")
            committed_rows.append((line, value))
        if committed_rows[-1][1] != last_key:
            raise ValueError(
                f"RNG progress mismatch: state ends at {last_key!r}, partial output ends at "
                f"{committed_rows[-1][1]!r}"
            )

        if len(partial_lines) > completed_rows:
            temporary = self.partial_path.with_suffix(self.partial_path.suffix + ".rollback.tmp")
            temporary.write_text(
                "".join(line if line.endswith("\n") else line + "\n" for line, _ in committed_rows),
                encoding="utf-8",
            )
            temporary.replace(self.partial_path)
        restore_rng_state(saved.get("rng_state", {}))
        return completed_rows, last_key

    def checkpoint_rng_progress(self, handle: TextIO, completed_rows: int, last_key: str) -> None:
        """Commit output rows and the matching post-generation RNG state."""

        if completed_rows <= 0 or not last_key:
            raise ValueError("RNG progress requires a positive row count and last row key.")
        handle.flush()
        os.fsync(handle.fileno())
        torch.save(
            {
                "format_version": 1,
                "completed_rows": completed_rows,
                "last_key": last_key,
                "rng_state": capture_rng_state(),
            },
            self.rng_temporary_path,
        )
        self.rng_temporary_path.replace(self.rng_state_path)

    def publish(self) -> None:
        if not self.partial_path.is_file():
            raise FileNotFoundError(f"No partial output to publish: {self.partial_path}")
        self.partial_path.replace(self.path)
        self.rng_state_path.unlink(missing_ok=True)
        self.rng_temporary_path.unlink(missing_ok=True)


def write_json_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
