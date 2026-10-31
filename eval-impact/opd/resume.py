"""Keep logs consistent with the checkpoint being resumed; archive unsaved work."""
import json
import os
import tempfile
from pathlib import Path


def rewind_log(path, step, archive_dir):
    path = Path(path)
    if not path.exists():
        return
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / path.name
    discarded = 0
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as kept:
        temporary = Path(kept.name)
        try:
            with path.open(encoding="utf-8") as source, archive.open("x", encoding="utf-8") as tail:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # An interrupted final write may leave an incomplete line.
                        if line.endswith("\n"):
                            raise
                        record = {"step": step + 1}
                    if record["step"] > step:
                        tail.write(line)
                        discarded += 1
                    else:
                        kept.write(line)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if discarded:
        os.replace(temporary, path)
    else:
        temporary.unlink()
        archive.unlink()
