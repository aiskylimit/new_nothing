"""Minimal data interface: a prompt set is a file of prompts, given by path in the config.

Accepted formats (chosen by extension):
    .txt    one prompt per line (blank lines skipped; a prompt cannot contain a newline)
    .json   a list of strings, or a list of objects with a "prompt" field
    .jsonl  one object per line with a "prompt" field

Objects may carry extra fields (e.g. "id", "behavior", "family"); fitting uses only the prompt text,
evaluation keeps the extra fields in its outputs. "behavior" is what the HarmBench judge compares the
response against (falls back to the prompt).
"""

import hashlib
import json
from pathlib import Path
from typing import List

from selot.config import resolve_path


def load_records(path) -> List[dict]:
    # Lines are split on "\n" only: str.splitlines() also splits on U+0085 / U+2028 / \x1c..., which can sit inside a
    # JSON string written with ensure_ascii=False (mmlu.jsonl has U+0085) and would cut a record in two.
    if not path:
        raise ValueError("a prompt-set path is empty: set it in the config (data.fit.* / data.eval[*].path)")
    path = resolve_path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        items = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").split("\n")]
        items = [s for s in items if s.strip()]
    elif suffix == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    else:
        raise ValueError(f"{path}: unsupported prompt file type {suffix!r} (use .txt, .json or .jsonl)")
    records = []
    for i, item in enumerate(items):
        rec = {"prompt": item} if isinstance(item, str) else dict(item)
        if not isinstance(rec.get("prompt"), str) or not rec["prompt"].strip():
            raise ValueError(f"{path}: item {i} has no non-empty 'prompt'")
        rec.setdefault("id", i)
        records.append(rec)
    if not records:
        raise ValueError(f"{path}: no prompts")
    return records


def load_prompts(path) -> List[str]:
    return [r["prompt"] for r in load_records(path)]


def prompts_hash(prompts: List[str]) -> str:
    return hashlib.sha256("\x00".join(prompts).encode("utf-8")).hexdigest()[:16]
