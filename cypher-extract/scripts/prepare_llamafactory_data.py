#!/usr/bin/env python3
"""Convert data/prepared into local LlamaFactory OpenAI-chat datasets."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from distillation.prepare_data import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
