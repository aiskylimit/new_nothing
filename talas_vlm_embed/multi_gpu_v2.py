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
DEFAULT_TARGET_POWER_W = 600.0  # Dùng cho GPU không được liệt kê ở trên

GPU_MAX_POWER_W = None

POWER_CHECK_INTERVAL = 0.2  # Giây, tần suất đo lại công suất & điều chỉnh duty
MICRO_CYCLE_S = 0.005       # Giây, độ dài mỗi micro-cycle PWM (càng nhỏ càng mượt)
DUTY_MIN = 0.02             # Duty cycle tối thiểu (không tắt hẳn để tránh dao động mạnh)
DUTY_MAX = 1.0              # Duty cycle tối đa (luôn tính, không nghỉ)
DUTY_STEP = 0.05            # Bước điều chỉnh duty cycle mỗi lần đo
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

    # Xác định mức công suất tối đa để hiển thị % (tự lấy TDP thật nếu không set cứng)
    if GPU_MAX_POWER_W is not None:
        gpu_max_power_w = GPU_MAX_POWER_W
    else:
        try:
            gpu_max_power_w = (
                pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            )
        except pynvml.NVMLError:
            gpu_max_power_w = target_power_w  # fallback an toàn nếu không đọc được

    print(
        f"[GPU {rank}] target={target_power_w:.0f}W "
        f"(TDP phát hiện ~{gpu_max_power_w:.0f}W)",
        flush=True
    )

    x = torch.randn(1524, 1524, device=device)
    y = torch.randn(1524, 1524, device=device)

    warmup_iters = 20
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(warmup_iters):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    per_mm_s = (time.time() - t0) / warmup_iters
    
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
                z = torch.mm(x, y)
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