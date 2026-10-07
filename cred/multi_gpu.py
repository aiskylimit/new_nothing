import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


import os
import time
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def find_free_port():
    """Yêu cầu OS cấp một port trống bất kỳ."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind vào port 0 nghĩa là xin OS chọn tự động 1 port đang trống
        s.bind(("", 0))
        # Trả về số port vừa được cấp
        return str(s.getsockname()[1])

def worker(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port  # Sử dụng port an toàn được truyền vào

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(1524, 1524, device=device)
    y = torch.randn(1524, 1524, device=device)

    while True:
        # GPU tính toán
        z = torch.mm(x, y)

        # Communication giữa TẤT CẢ GPU
        dist.all_reduce(z, op=dist.ReduceOp.SUM)

        # Đồng bộ
        torch.cuda.synchronize()

        time.sleep(0.01)

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("Không tìm thấy GPU CUDA")

    print(f"Detected {num_gpus} GPU")

    free_port = find_free_port()
    print(f"Khởi tạo DDP với MASTER_PORT = {free_port}")

    mp.spawn(
        worker,
        args=(num_gpus, free_port), # Truyền thêm free_port vào args cho hàm worker
        nprocs=num_gpus,
        join=True
    )