"""Khung train dùng chung cho cả 3 phương pháp.

Cố ý dùng vòng lặp torch thuần thay vì ``Trainer``/``SFTTrainer``: loss của ta
là hàm của **step span** trong từng mẫu và (ở Phase 1) cần nhiều forward trên
cùng một batch, hai thứ mà Trainer API không diễn đạt gọn được.

Ghi chú lệch so với spec §5: spec ghi DeepSpeed ZeRO-2. Với 1 GPU, base 7B đóng
băng và chỉ LoRA/gate được train thì ZeRO-2 (shard optimizer state) không giảm
được gì đáng kể — optimizer state chỉ vài chục MB. Ta chạy single-process
AdamW; ``scripts/`` vẫn để chỗ cắm ``accelerate``/``deepspeed`` nếu cần scale
sang nhiều GPU.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch



def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="YAML config (CLI override YAML)")
    parser.add_argument("--model_name", default=None, help="Bắt buộc (CLI hoặc YAML), không có model mặc định")
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--out_dir", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--global_batch", type=int, default=32,
                        help="Mô phỏng bằng gradient accumulation (micro batch = 1)")
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--segmentation_mode", default="paragraph",
                        choices=["paragraph", "sentence"])
    parser.add_argument("--loss_reduction", default="token_mean",
                        choices=["token_mean", "step_mean", "sum"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0,
                        help="Lưu checkpoint mỗi K optimizer step (0 = không lưu theo step)")
    parser.add_argument("--save_epochs", action="store_true", default=True,
                        help="Tự động lưu checkpoint ở cuối mỗi epoch")
    parser.add_argument("--no_save_epochs", dest="save_epochs", action="store_false")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Tự động tìm và tiếp tục từ checkpoint mới nhất trong out_dir")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Đường dẫn trực tiếp đến thư mục checkpoint cần resume")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", type=int, default=0,
                        help=">0: chỉ chạy N optimizer step rồi thoát (smoke test)")
    return parser


def find_latest_checkpoint(out_dir: Optional[str]) -> Optional[str]:
    """Tìm thư mục checkpoint mới nhất trong ``out_dir``."""
    import re
    if not out_dir or not os.path.isdir(out_dir):
        return None
    last_link = os.path.join(out_dir, "checkpoint-last")
    if os.path.isdir(last_link) and os.path.isfile(os.path.join(last_link, "trainer_state.pt")):
        return last_link

    candidates = []
    for entry in os.listdir(out_dir):
        full = os.path.join(out_dir, entry)
        if not os.path.isdir(full):
            continue
        if not os.path.isfile(os.path.join(full, "trainer_state.pt")):
            continue
        m_step = re.match(r"^checkpoint-(?:step-)?(\d+)$", entry)
        if m_step:
            candidates.append((int(m_step.group(1)), os.path.getmtime(full), full))
            continue
        m_ep = re.match(r"^checkpoint-epoch-(\d+)$", entry)
        if m_ep:
            candidates.append((int(m_ep.group(1)) * 100000, os.path.getmtime(full), full))
            continue
        if entry.startswith("checkpoint"):
            candidates.append((0, os.path.getmtime(full), full))

    if not candidates:
        return None
    # Ưu tiên checkpoint được GHI GẦN NHẤT (mtime), rồi mới tới số step/epoch.
    # Nhờ vậy checkpoint-step-200 mới hơn sẽ thắng checkpoint-epoch-1 cũ.
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return candidates[0][2]


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float):
    params = [p for p in params if p.requires_grad]
    if not params:
        raise RuntimeError("Không có tham số trainable — kiểm tra freeze/requires_grad")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


def cosine_schedule_with_warmup(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class Accumulator:
    """Gom metric giữa các micro-batch để log theo optimizer step."""

    values: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            self.values[key] = self.values.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: self.values[k] / max(self.counts[k], 1) for k in self.values}

    def reset(self) -> None:
        self.values.clear()
        self.counts.clear()


class StepTimer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start


def total_optimizer_steps(n_examples: int, epochs: int, global_batch: int) -> int:
    per_epoch = max(n_examples // global_batch, 1)
    return per_epoch * epochs


def save_run_config(out_dir: str, args: argparse.Namespace, extra: Optional[dict] = None) -> None:
    """Lưu lại toàn bộ hyperparameter + seed (spec §14: determinism)."""
    from core.data_utils import write_json

    payload = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    payload.update(extra or {})
    payload["torch_version"] = torch.__version__
    payload["cuda"] = torch.cuda.is_available()
    write_json(os.path.join(out_dir, "run_config.json"), payload)


def clip_and_step(params: List[torch.nn.Parameter], optimizer, scheduler, max_grad_norm: float) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)
