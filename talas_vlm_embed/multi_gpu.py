import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29505"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(512, 512, device=device)
    y = torch.randn(512, 512, device=device)

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

    mp.spawn(
        worker,
        args=(num_gpus,),
        nprocs=num_gpus,
        join=True
    )