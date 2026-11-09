"""Đặt tên tiến trình hiển thị trong nvidia-smi / ps (tiền tố PROC_PREFIX, mặc định hieunq10).

nvidia-smi chỉ hiện ~32 ký tự cuối của tên nên giữ tiền tố + phần đuôi (phần phân biệt nhất).
Gọi sớm trong main(), trước khi import vllm, để tiến trình con VLLM::EngineCore cũng mang tiền tố.
"""
from __future__ import annotations

import os

_MAX_LEN = 32


def set_proc_title(suffix: str) -> str:
    prefix = os.environ.get("PROC_PREFIX", "hieunq10")
    os.environ.setdefault("VLLM_PROCESS_NAME_PREFIX", prefix)
    head = f"{prefix}-"
    name = head + suffix
    if len(name) > _MAX_LEN:
        name = head + suffix[-(_MAX_LEN - len(head)):]
    try:
        import setproctitle
        setproctitle.setproctitle(name)
    except ImportError:
        pass
    return name
