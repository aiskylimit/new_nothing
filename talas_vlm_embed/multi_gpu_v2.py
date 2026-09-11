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

# ============ CẤU HÌNH MỤC TIÊU CÔNG SUẤT ============
TARGET_POWER_W = 600.0      # Công suất mục tiêu (W) trên mỗi GPU
GPU_MAX_POWER_W = 1000.0    # Công suất tối đa (TDP) của GPU, chỉ để hiển thị %
POWER_CHECK_INTERVAL = 0.2  # Giây, tần suất đo lại công suất để điều chỉnh
DUTY_MIN = 0.02             # Duty cycle tối thiểu (tránh vòng lặp busy 100% CPU khi idle)
DUTY_MAX = 1.0              # Duty cycle tối đa (luôn tính, không nghỉ)
DUTY_STEP = 0.05            # Bước điều chỉnh duty cycle mỗi lần đo
# ======================================================


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

    x = torch.randn(1524, 1524, device=device)
    y = torch.randn(1524, 1524, device=device)

    # duty = tỉ lệ thời gian "bận tính toán" trên mỗi chu kỳ điều khiển
    duty = 0.5  # bắt đầu ở giữa, feedback loop sẽ tự điều chỉnh
    last_check = time.time()

    step_count = 0

    while True:
        cycle_start = time.time()
        # Thời gian "bận" trong chu kỳ này, dựa trên duty hiện tại
        busy_deadline = cycle_start + duty * POWER_CHECK_INTERVAL

        # Vòng lặp tính toán cho tới khi hết thời gian "bận" trong chu kỳ
        while time.time() < busy_deadline:
            z = torch.mm(x, y)
            dist.all_reduce(z, op=dist.ReduceOp.SUM)
            step_count += 1

        torch.cuda.synchronize()

        # Phần còn lại của chu kỳ: nghỉ (nếu duty < 1.0)
        remaining = POWER_CHECK_INTERVAL - (time.time() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)

        # Đo công suất thực tế và điều chỉnh duty cycle (feedback control)
        now = time.time()
        if now - last_check >= POWER_CHECK_INTERVAL:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatt
            power_w = power_mw / 1000.0

            error = TARGET_POWER_W - power_w
            # Điều chỉnh duty theo sai số, giới hạn trong [DUTY_MIN, DUTY_MAX]
            if error > 0:
                duty = min(DUTY_MAX, duty + DUTY_STEP)
            elif error < 0:
                duty = max(DUTY_MIN, duty - DUTY_STEP)

            last_check = now

            if rank == 0:
                pct = 100.0 * power_w / GPU_MAX_POWER_W
                print(
                    f"[GPU {rank}] power={power_w:.1f}W "
                    f"({pct:.1f}% of {GPU_MAX_POWER_W:.0f}W) "
                    f"target={TARGET_POWER_W:.0f}W duty={duty:.2f}",
                    flush=True
                )


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("Không tìm thấy GPU CUDA")

    print(f"Detected {num_gpus} GPU")
    print(f"Mục tiêu công suất: {TARGET_POWER_W}W / {GPU_MAX_POWER_W}W mỗi GPU")
    free_port = find_free_port()
    print(f"Khởi tạo DDP với MASTER_PORT = {free_port}")

    mp.spawn(
        worker,
        args=(num_gpus, free_port),
        nprocs=num_gpus,
        join=True
    )