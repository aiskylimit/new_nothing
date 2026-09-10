import torch
import torch.nn as nn


class ETP(nn.Module):
    def __init__(
        self,
        sinkhorn_alpha: float = 0.1,
        stop_threshold: float = 1e-9,
        max_iter: int = 100,
        epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.stop_threshold = stop_threshold
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cost.ndim != 2:
            raise ValueError(f"ETP expects a 2D cost matrix, got shape {tuple(cost.shape)}")
        if cost.shape[0] == 0 or cost.shape[1] == 0:
            zero = cost.new_zeros(())
            return zero, cost.new_zeros(cost.shape)

        original_dtype = cost.dtype
        cost = cost.float()
        rows, cols = cost.shape
        a = cost.new_full((rows, 1), 1.0 / rows)
        b = cost.new_full((cols, 1), 1.0 / cols)
        u = cost.new_full((rows, 1), 1.0 / rows)

        kernel = torch.exp(-cost * self.sinkhorn_alpha).clamp_min(self.epsilon)
        step = 0
        while step < self.max_iter:
            v = b / (kernel.t().matmul(u) + self.epsilon)
            u = a / (kernel.matmul(v) + self.epsilon)
            step += 1
            if step % 50 == 1:
                marginal = v * kernel.t().matmul(u)
                err = torch.norm(torch.sum(torch.abs(marginal - b), dim=0), p=float("inf"))
                # The old while condition copied this unchanged CUDA scalar to
                # the CPU on every iteration.  It only changes at checkpoints.
                if not bool(err > self.stop_threshold):
                    break

        transport = u * (kernel * v.t())
        loss = torch.sum(transport * cost)
        return loss.to(dtype=original_dtype), transport.to(dtype=original_dtype)
