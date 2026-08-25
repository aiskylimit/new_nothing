import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornCentroidDescriptor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_centroids=64,
        centroid_dim=128,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=5,
        dustbin_mass=None,
    ):
        super().__init__()
        if dustbin_mass is not None and not 0.0 < dustbin_mass < 1.0:
            raise ValueError(f"dustbin_mass must be in (0, 1), got {dustbin_mass}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_centroids = num_centroids
        self.centroid_dim = centroid_dim
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.dustbin_mass = dustbin_mass

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.centroids = nn.Parameter(torch.empty(num_centroids + 1, hidden_dim))
        self.down_proj = nn.Linear(hidden_dim, centroid_dim)
        self.reset_parameters()

    @property
    def descriptor_dim(self):
        return self.num_centroids * self.centroid_dim

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.centroids, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    def _target_log_marginals(self, token_mask):
        bsz, seq_len = token_mask.shape
        device = token_mask.device
        dtype = torch.float32

        num_valid = token_mask.sum(dim=1).clamp(min=1).to(dtype)
        token_mass = token_mask.to(dtype) / num_valid.unsqueeze(1)
        log_mu = torch.where(
            token_mask,
            token_mass.clamp_min(1e-8).log(),
            torch.full((bsz, seq_len), -1e4, device=device, dtype=dtype),
        )

        if self.dustbin_mass is None:
            dustbin_mass = (num_valid - self.num_centroids).clamp_min(0.0) / (num_valid + self.num_centroids)
            dustbin_mass = dustbin_mass.clamp_min(1e-6)
        else:
            dustbin_mass = torch.full((bsz,), self.dustbin_mass, device=device, dtype=dtype)

        nu = torch.empty(bsz, self.num_centroids + 1, device=device, dtype=dtype)
        nu[:, : self.num_centroids] = (1.0 - dustbin_mass).unsqueeze(1) / self.num_centroids
        nu[:, -1] = dustbin_mass
        return log_mu, nu.log()

    def sinkhorn(self, logits, token_mask):
        log_alpha = logits.float() / self.sinkhorn_epsilon
        log_alpha = log_alpha.masked_fill(~token_mask.unsqueeze(-1), -1e4)
        log_mu, log_nu = self._target_log_marginals(token_mask)

        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(self.sinkhorn_iters):
            u = log_mu - torch.logsumexp(log_alpha + v.unsqueeze(1), dim=-1)
            v = log_nu - torch.logsumexp(log_alpha + u.unsqueeze(-1), dim=1)

        transport = (log_alpha + u.unsqueeze(-1) + v.unsqueeze(1)).exp()
        return transport.masked_fill(~token_mask.unsqueeze(-1), 0.0)

    def forward(self, token_hidden, token_mask):
        token_hidden = self.input_proj(token_hidden.float())
        centroids = self.centroids.float()

        logits = -torch.cdist(token_hidden, centroids, p=2).pow(2)
        transport = self.sinkhorn(logits, token_mask)

        assignment = transport[:, :, : self.num_centroids]
        mass = assignment.sum(dim=1).clamp_min(1e-8)
        residual = token_hidden.unsqueeze(2) - centroids[: self.num_centroids].view(1, 1, self.num_centroids, -1)
        centroid_tokens = torch.einsum("bnk,bnkd->bkd", assignment, residual)
        centroid_tokens = centroid_tokens / mass.unsqueeze(-1)

        slots = self.down_proj(centroid_tokens)
        descriptor = slots.flatten(start_dim=1)
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        return descriptor, {
            "transport": transport,
            "centroid_mass": mass.detach(),
            "dustbin_mass": transport[:, :, -1].sum(dim=1).detach(),
        }
