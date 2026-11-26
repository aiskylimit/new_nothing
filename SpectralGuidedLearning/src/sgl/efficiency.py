"""Wall time and peak GPU memory of every executed stage, for efficiency reporting.

Opt-in per config: `run.efficiency_dir` appends one JSON line per stage run to
<efficiency_dir>/<run_name>.jsonl. sgl.diagnostics.efficiency turns those into a table.
"""

import json
import subprocess
import threading
import time
from pathlib import Path


def gpu_memory_mib(gpus: str | None) -> int | None:
    """Memory in use summed over `gpus` (CUDA_VISIBLE_DEVICES syntax); None without nvidia-smi."""
    command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if gpus:
        command += ["-i", gpus]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout
        return sum(int(line) for line in output.split())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class StageMeter:
    """Times a stage and samples the memory of its GPUs while it runs.

    The sample is device-wide: `start_gpu_mib` shows what other processes already held, and a vLLM
    stage reports what it reserved (gpu-memory-utilization), not what it needed.
    """

    def __init__(self, gpus: str | None, interval: float = 2.0):
        self.gpus, self.interval = gpus, interval
        self._stop = threading.Event()

    def __enter__(self):
        self.start_gpu_mib = self.peak_gpu_mib = gpu_memory_mib(self.gpus)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        if self.start_gpu_mib is not None:
            self._thread.start()
        self._started = time.monotonic()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval):
            used = gpu_memory_mib(self.gpus)
            if used is not None:
                self.peak_gpu_mib = max(self.peak_gpu_mib, used)

    def __exit__(self, *exc):
        self.seconds = time.monotonic() - self._started
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        return False


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
