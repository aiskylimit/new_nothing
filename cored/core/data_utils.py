"""Dataset + config + seeding dùng chung cho mọi script train.

Toàn bộ pipeline train chạy ở **per-device batch size 1** và mô phỏng global
batch 32 bằng gradient accumulation. Lý do: mỗi mẫu có số step N khác nhau và
step span là chỉ số token tuyệt đối, padding theo batch sẽ làm rối alignment —
đây đúng là chỗ spec §14 cảnh báo dễ bug nhất.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from .segmentation import SegmentedExample, build_segmented_example


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class TrainExample:
    """Một mẫu đã tokenize + span, sẵn sàng đưa vào loss."""

    example_id: str
    segmented: SegmentedExample
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_ids(self) -> torch.Tensor:
        return torch.tensor(self.segmented.input_ids, dtype=torch.long)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(self.segmented.labels, dtype=torch.long)

    @property
    def step_spans(self):
        return self.segmented.step_spans

    @property
    def answer_span(self):
        return self.segmented.answer_span


class StepDataset(torch.utils.data.Dataset):
    """Đọc JSONL từ ``data/prepare.py`` và tokenize theo yêu cầu."""

    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        cutoff_len: int = 4096,
        segmentation_mode: str = "paragraph",
        limit: Optional[int] = None,
        min_steps: int = 2,
    ):
        self.rows = read_jsonl(path, limit=limit)
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.segmentation_mode = segmentation_mode
        self.min_steps = min_steps

    def __len__(self) -> int:
        return len(self.rows)

    def example_id(self, index: int) -> str:
        row = self.rows[index]
        return str(row.get("id", index))

    def __getitem__(self, index: int) -> TrainExample:
        row = self.rows[index]
        segmented = build_segmented_example(
            self.tokenizer,
            question=row["question"],
            thinking=row.get("thinking") or "\n\n".join(row["steps"]),
            answer=row["answer_text"] if "answer_text" in row else row.get("answer", ""),
            segmentation_mode=self.segmentation_mode,
            cutoff_len=self.cutoff_len,
        )
        return TrainExample(example_id=self.example_id(index), segmented=segmented, raw=row)

    def iter_epoch(self, seed: int, epoch: int) -> Iterator[int]:
        order = list(range(len(self)))
        random.Random(seed + epoch).shuffle(order)
        return iter(order)


# --------------------------------------------------------------------------- #
# Seeding / determinism (spec §14)
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# --------------------------------------------------------------------------- #
# Config: YAML + CLI override
# --------------------------------------------------------------------------- #
def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_config(args, config_key: str = "config"):
    """Merge YAML vào namespace argparse: CLI > YAML > default.

    Chỉ ghi đè field mà người dùng **không** truyền trên CLI (nhận biết bằng
    cách so với default của parser, do caller gắn vào ``args._cli_provided``).
    """
    config = load_yaml(getattr(args, config_key, None))
    provided = getattr(args, "_cli_provided", set())
    for key, value in config.items():
        if key in provided:
            continue
        setattr(args, key, value)
    return args


def track_cli_provided(parser, argv: Optional[Sequence[str]] = None):
    """Trả về namespace + tập tên arg thực sự có mặt trên CLI."""
    import sys

    argv = list(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(argv)
    provided = set()
    for token in argv:
        if token.startswith("--"):
            provided.add(token[2:].split("=")[0].replace("-", "_"))
    args._cli_provided = provided
    return apply_config(args)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class JsonlLogger:
    """Ghi metric từng step ra ``<out_dir>/train_log.jsonl``."""

    def __init__(self, out_dir: str, filename: str = "train_log.jsonl"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)
        self.handle = open(self.path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
