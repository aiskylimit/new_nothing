from __future__ import annotations

from pathlib import Path
from typing import Any


def create_reference_model_at_revision(
    model_args: Any,
    finetuning_args: Any,
    ref_model_revision: str | None,
    factory: Any,
) -> Any:
    """Create a remote or local teacher without leaking the student's revision."""

    student_revision = model_args.model_revision
    try:
        if ref_model_revision is not None:
            model_args.model_revision = ref_model_revision
        elif Path(str(getattr(finetuning_args, "ref_model", ""))).expanduser().is_dir():
            # Local full-model checkpoints have no Hub revision. LlamaFactory
            # otherwise forwards the student's pinned revision to the teacher.
            model_args.model_revision = None
        return factory(model_args, finetuning_args)
    finally:
        model_args.model_revision = student_revision
