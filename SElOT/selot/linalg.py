"""Float32 linear-algebra helpers used by every fit (PCA, symmetric matrix roots, weighted ridge)."""

import torch

DTYPE = torch.float32


def as_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(DTYPE)


def sym(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.T)


def sym_power(a: torch.Tensor, power: float) -> torch.Tensor:
    """a^power for a symmetric PSD matrix (eigenvalues clamped at 0; at a tiny floor for negative powers)."""
    evals, evecs = torch.linalg.eigh(sym(as_f32(a)))
    floor = 0.0 if power > 0 else torch.finfo(DTYPE).eps * evals.abs().max()
    evals = evals.clamp_min(floor)
    return sym((evecs * evals.pow(power)) @ evecs.T)


def norm_weights(w, n: int, device) -> torch.Tensor:
    """Sample weights summing to 1; None = uniform 1/n."""
    if w is None:
        return torch.full((n,), 1.0 / n, dtype=DTYPE, device=device)
    w = as_f32(torch.as_tensor(w)).to(device)
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"sample weights must be {n} non-negative numbers with a positive sum")
    return w / w.sum()


def wmean(x: torch.Tensor, w=None) -> torch.Tensor:
    x = as_f32(x)
    return x.mean(0) if w is None else norm_weights(w, x.shape[0], x.device) @ x


def mean_cov(x: torch.Tensor, eps: float = 0.0, w=None):
    """Mean and unbiased covariance of the rows of x, covariance shrunk as S + eps I.

    With sample weights w: weighted mean and the reliability-weighted unbiased covariance
    sum_i w_i (x_i - mu)(x_i - mu)^T / (1 - sum_i w_i^2), w normalised to sum 1 (equal weights give the plain one).
    """
    x = as_f32(x)
    if w is None:
        mu = x.mean(0)
        xc = x - mu
        cov = xc.T @ xc / max(x.shape[0] - 1, 1)
    else:
        w = norm_weights(w, x.shape[0], x.device)
        mu = w @ x
        xc = x - mu
        cov = xc.T @ (w[:, None] * xc) / (1 - (w ** 2).sum()).clamp_min(1e-12)
    return mu, sym(cov) + eps * torch.eye(x.shape[1], dtype=DTYPE, device=x.device)


def pca_basis(x_centered: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r right singular vectors (d x r, orthonormal columns) of an already centered n x d matrix.

    Uses the eigen-decomposition of the smaller Gram matrix, so it stays cheap for n << d and n >> d.
    """
    x = as_f32(x_centered)
    n, d = x.shape
    r = min(r, n, d)
    if n < d:
        evals, evecs = torch.linalg.eigh(x @ x.T)                 # ascending
        evals, evecs = evals.flip(0)[:r], evecs.flip(1)[:, :r]
        basis = x.T @ evecs / evals.clamp_min(torch.finfo(DTYPE).tiny).sqrt()
        basis, _ = torch.linalg.qr(basis)                         # re-orthonormalize in float32
        return basis
    evals, evecs = torch.linalg.eigh(x.T @ x)
    return evecs.flip(1)[:, :r].contiguous()


def weighted_ridge(features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """argmin_M  sum_i w_i ||M f_i - t_i||^2 + gamma ||M||_F^2  =  T Omega F^T (F Omega F^T + gamma I)^-1.

    features: n x p (rows are samples), targets: n x k, weights: n. Returns M: k x p.
    """
    f, t, w = as_f32(features), as_f32(targets), as_f32(weights)
    gram = f.T @ (w[:, None] * f)
    gram = sym(gram) + gamma * torch.eye(f.shape[1], dtype=DTYPE, device=f.device)
    rhs = f.T @ (w[:, None] * t)                                  # p x k
    try:
        chol = torch.linalg.cholesky(gram)
        sol = torch.cholesky_solve(rhs, chol)
    except RuntimeError:                                          # gamma = 0 and rank-deficient: least norm
        sol = torch.linalg.lstsq(gram, rhs, driver="gelsd" if gram.device.type == "cpu" else None).solution
    return sol.T.contiguous()
