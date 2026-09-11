import os
import time
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import pynvml
except ImportError:
    raise ImportError(
        "Cần cài pynvml để đo công suất GPU: pip install nvidia-ml-py3"
    )

# ============ CẤU HÌNH MỤC TIÊU CÔNG SUẤT (THEO TỪNG GPU) ============
TARGET_POWER_PER_GPU_W = {
    # 0: 500.0,
    # 1: 300.0,
    # 2: 700.0,
    # 3: 700.0,
}
DEFAULT_TARGET_POWER_W = 900.0  # Dùng cho GPU không được liệt kê ở trên

GPU_MAX_POWER_W = None

POWER_CHECK_INTERVAL = 0.2  # Giây, tần suất đo lại công suất & điều chỉnh duty
MICRO_CYCLE_S = 0.02        # Giây, độ dài mỗi micro-cycle PWM (càng nhỏ càng mượt)
DUTY_MIN = 0.02             # Duty cycle tối thiểu (không tắt hẳn để tránh dao động mạnh)
DUTY_MAX = 1.0              # Duty cycle tối đa (luôn tính, không nghỉ)
DUTY_STEP = 0.05            # Bước điều chỉnh duty cycle mỗi lần đo

# --- Giới hạn VRAM cho ma trận burn (KHÔNG dùng đa luồng/nhiều stream) ---
MAX_VRAM_GB = 6.0          # Tổng VRAM tối đa dùng cho x, y, z (fp32)
NUM_MATRICES = 3           # x, y, z (out buffer) mỗi cái là 1 ma trận NxN
BYTES_PER_ELEM = 4         # fp32
MIN_MM_TIME_S = 0.002      # 1 lần mm nên tốn tối thiểu ~2ms để overhead không đáng kể
MIN_MATRIX_SIZE = 1524     # kích thước khởi điểm để dò
# ======================================================================


def find_free_port():
    """Yêu cầu OS cấp một port trống bất kỳ."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def worker(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Khởi tạo NVML để đo công suất thực tế của GPU này
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)

    # Target công suất riêng cho GPU này (fallback về default nếu không cấu hình)
    target_power_w = TARGET_POWER_PER_GPU_W.get(rank, DEFAULT_TARGET_POWER_W)

    # --- Tính kích thước ma trận tối đa cho phép trong ngân sách VRAM ---
    max_vram_bytes = MAX_VRAM_GB * (1024 ** 3)
    max_elems = max_vram_bytes / (NUM_MATRICES * BYTES_PER_ELEM)
    hard_cap_size = int(max_elems ** 0.5)

    matrix_size = min(MIN_MATRIX_SIZE, hard_cap_size)
    per_mm_s = None

    while True:
        x = torch.randn(matrix_size, matrix_size, device=device)
        y = torch.randn(matrix_size, matrix_size, device=device)
        z = torch.empty(matrix_size, matrix_size, device=device)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            torch.mm(x, y, out=z)
        torch.cuda.synchronize()
        per_mm_s = (time.time() - t0) / 5

        if per_mm_s >= MIN_MM_TIME_S or matrix_size >= hard_cap_size:
            break

        del x, y, z
        torch.cuda.empty_cache()
        matrix_size = min(matrix_size * 2, hard_cap_size)

    max_iters_per_micro = max(1, int(MICRO_CYCLE_S / per_mm_s))

    duty = 0.5  # bắt đầu ở giữa, feedback loop sẽ tự điều chỉnh
    last_check = time.time()

    step_count = 0

    while True:
        cycle_start = time.time()

        while time.time() - cycle_start < POWER_CHECK_INTERVAL:
            micro_start = time.time()
            n_iters = max(1, round(duty * max_iters_per_micro))

            for _ in range(n_iters):
                torch.mm(x, y, out=z)
            torch.cuda.synchronize()  # đảm bảo GPU thực sự rảnh trước khi sleep
            step_count += n_iters

            elapsed = time.time() - micro_start
            remaining_micro = MICRO_CYCLE_S - elapsed
            if remaining_micro > 0:
                time.sleep(remaining_micro)

        dist.all_reduce(z, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Đo công suất thực tế và điều chỉnh duty cycle (feedback control)
        now = time.time()
        if now - last_check >= POWER_CHECK_INTERVAL:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatt
            power_w = power_mw / 1000.0

            error = target_power_w - power_w
            # Điều chỉnh duty theo sai số, giới hạn trong [DUTY_MIN, DUTY_MAX]
            if error > 0:
                duty = min(DUTY_MAX, duty + DUTY_STEP)
            elif error < 0:
                duty = max(DUTY_MIN, duty - DUTY_STEP)

            last_check = now


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("Không tìm thấy GPU CUDA")

    free_port = find_free_port()
    print(f"Khởi tạo DDP với MASTER_PORT = {free_port}")

    mp.spawn(
        worker,
        args=(num_gpus, free_port),
        nprocs=num_gpus,
        join=True
    )