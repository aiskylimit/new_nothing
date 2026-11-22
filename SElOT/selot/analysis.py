"""Activation-level analyses (proposal Sec. 4.4), computed from unsteered activations and the fitted map.

RQ3  How far does the map move each group? ||A h + b - h|| = alpha ||g(h)|| per prompt group, at the last
     prompt token (where the field was fit) and at every prompt position (where it is also applied).
(i)  Does transport match the refusal distribution? In the transport subspace P (SELOT / PCA-OT maps keep
     it in extras): Gaussian W2 and covariance alignment of jailbreaks vs refused, before and after.
"""

import torch

from selot.affine_map import LayerMap
from selot.gaussian_ot import covariance_alignment, gaussian_w2_sq
from selot.linalg import mean_cov


def norm_stats(x: torch.Tensor) -> dict:
    n = x.norm(dim=-1)
    return {"n": int(n.numel()), "mean": float(n.mean()), "median": float(n.median()),
            "p95": float(torch.quantile(n, 0.95)) if n.numel() > 1 else float(n.max()), "max": float(n.max())}


def displacement_stats(lmap: LayerMap, h: torch.Tensor, alpha: float) -> dict:
    disp = alpha * lmap.displacement(h)
    stats = norm_stats(disp)
    stats["relative_mean"] = float((disp.norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-12)).mean())
    return stats


def transport_match(lmap: LayerMap, hj: torch.Tensor, hr: torch.Tensor, alpha: float, eps: float = 1e-4):
    if "P" not in lmap.extras:
        return None
    p, mu = lmap.extras["P"], lmap.extras["mu_pool"]
    proj = lambda h: (h.float() - mu) @ p
    m_r, s_r = mean_cov(proj(hr), eps)
    out = {}
    for tag, h in (("before", hj), ("after", lmap.apply(hj, alpha))):
        m, s = mean_cov(proj(h), eps)
        out[tag] = {"w2_sq": gaussian_w2_sq(m, s, m_r, s_r), "cov_alignment": covariance_alignment(s, s_r),
                    "mean_dist": float((m - m_r).norm())}
    return out
