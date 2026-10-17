"""Shared locations for assets downloaded onto the training server."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "CYPHER_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(
    "/mnt/local/aiskylimit_new_nothing/cypher-extract/datasets/cypher-extract-data"
)


def get_data_root() -> Path:
    """Return the downloaded dataset root, allowing an explicit override."""

    configured = os.environ.get(DATA_ROOT_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_DATA_ROOT
