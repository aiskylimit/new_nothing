"""Closed-form OT map between two Gaussians (proposal Eq. 4)."""

import torch

from selot.linalg import as_f32, sym, sym_power


def gaussian_ot_map(m1: torch.Tensor, s1: torch.Tensor, m2: torch.Tensor, s2: torch.Tensor):
    """T(x) = A x + b pushes N(m1, S1) onto N(m2, S2).

    A = S1^{-1/2} (S1^{1/2} S2 S1^{1/2})^{1/2} S1^{-1/2},  b = m2 - A m1.
    """
    m1, s1, m2, s2 = map(as_f32, (m1, s1, m2, s2))
    s1_half = sym_power(s1, 0.5)
    s1_inv_half = sym_power(s1, -0.5)
    middle = sym_power(s1_half @ s2 @ s1_half, 0.5)
    a = sym(s1_inv_half @ middle @ s1_inv_half)
    b = m2 - a @ m1
    return a, b


def gaussian_w2_sq(m1, s1, m2, s2) -> float:
    """Squared 2-Wasserstein distance between N(m1, S1) and N(m2, S2)."""
    m1, s1, m2, s2 = map(as_f32, (m1, s1, m2, s2))
    s1_half = sym_power(s1, 0.5)
    cross = sym_power(s1_half @ s2 @ s1_half, 0.5)
    return float(((m1 - m2) ** 2).sum() + torch.trace(s1 + s2 - 2 * cross))


def covariance_alignment(s1, s2) -> float:
    """<S1, S2>_F / (||S1||_F ||S2||_F): 1 means the two covariances have the same shape."""
    s1, s2 = as_f32(s1), as_f32(s2)
    return float((s1 * s2).sum() / (s1.norm() * s2.norm()).clamp_min(1e-30))


def check_ot_map(a, b, m1, s1, m2, s2) -> dict:
    """Relative errors of the two defining identities: A m1 + b = m2 and A S1 A = S2."""
    mean_err = (a @ m1 + b - m2).norm() / m2.norm().clamp_min(1e-12)
    cov_err = (a @ s1 @ a.T - s2).norm() / s2.norm().clamp_min(1e-12)
    evals = torch.linalg.eigvalsh(sym(a))
    return {"ot_mean_rel_err": float(mean_err), "ot_cov_rel_err": float(cov_err),
            "ot_A_min_eig": float(evals.min()), "ot_A_asym": float((a - a.T).norm())}
