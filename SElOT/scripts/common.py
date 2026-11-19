"""Shared setup for the stage scripts: import path, logging, config, JSON helpers."""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selot.config import add_config_args, load_config, run_paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def setup(description: str):
    parser = add_config_args(argparse.ArgumentParser(description=description))
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    paths = run_paths(cfg)
    paths["results"].mkdir(parents=True, exist_ok=True)
    return cfg, paths, logging.getLogger(Path(sys.argv[0]).stem)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
