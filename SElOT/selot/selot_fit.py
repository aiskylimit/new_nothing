"""SELOT fit at one layer (proposal Sec. 3.2, Algorithm 1), all in float32.

Inputs are last-prompt-token activations of the unmodified model, as rows:
    Hb (benign, n_b x d), Hr (refused harmful, n_r x d), Hj (jailbreaks, n_j x d).

Steps
    (b) mu_pool = (n_j mu_j + n_r mu_r)/(n_j + n_r);  P = top-k right singular vectors of [Hj; Hr] - mu_pool
    (c) y = P^T (h - mu_pool);  (A_k, b_k) = Gaussian OT from N(m_j, S_j + eps I) to N(m_r, S_r + eps I)
        delta*(y) = (A_k - I) y + b_k  for every jailbreak                                   (Eq. 6)
    (d) psi(h) = [Q^T (h - mu_all); 1]                                                       (Eq. 7)
        soft:  M* = T Omega Psi^T (Psi Omega Psi^T + gamma I)^-1                             (Eq. 10)
               weights 1/n_j (jailbreak, target delta*), beta/n_b (benign, target 0), beta_r/n_r (refused, 0)
        hard:  M = M~ N_b, N_b = projector on the small-eigenvalue eigenvectors of Psi_b Psi_b^T;
               M~ fit on Dj and Dr only
    g(h) = P M psi(h) = U V^T h + b0  with  U = P,  V = Q W^T,  b0 = P (c - W Q^T mu_all),  M = [W | c]

Sample weights (optional) w_r, w_j on the refused and jailbreak rows (inverse topic frequency): every mean,
covariance and PCA over Hr / Hj is weighted, and the ridge weights 1/n_j, beta_r/n_r become w_j, beta_r w_r
(normalised to sum 1). No weights = the unweighted fit above.

Selector options (Table 4): "topm" (Q = top-m PCs of all three sets), "y" (Q = P, mu_all = mu_pool,
i.e. psi = [y; 1]), "full" (Q = I). Target options: "ot" (delta*) or "constant" (m_r - m_j for every
jailbreak, the RQ2 ablation).
"""

from dataclasses import dataclass

import torch

from selot.affine_map import LayerMap
from selot.gaussian_ot import check_ot_map, gaussian_ot_map
from selot.linalg import DTYPE, as_f32, mean_cov, norm_weights, pca_basis, sym, weighted_ridge, wmean


@dataclass
class SelotParams:
    k: int = 5                  # transport subspace size
    m: int = 256                # selector feature size (selector="topm")
    beta: float = 100.0         # benign cost
    beta_r: float = 1.0         # refused-harmful cost
    gamma: float = 1e-3         # ridge
    eps: float = 1e-4           # covariance shrinkage
    variant: str = "soft"       # soft | hard
    selector: str = "topm"      # topm | y | full
    target: str = "ot"          # ot | constant
    null_rel_tol: float = 1e-3  # hard variant: eigenvalue < tol * max eigenvalue counts as null
    min_null_ratio: float = 0.0 # hard variant: keep at least this fraction of the m+1 directions

    @classmethod
    def from_dict(cls, d: dict) -> "SelotParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def transport_subspace(hj, hr, k, wj=None, wr=None):
    nj, nr = hj.shape[0], hr.shape[0]
    mu_j, mu_r = wmean(hj, wj), wmean(hr, wr)
    mu_pool = (nj * mu_j + nr * mu_r) / (nj + nr)
    x = torch.cat([hj, hr]) - mu_pool
    if wj is not None or wr is not None:      # row i scaled by sqrt(n w_i): equal weights leave the rows unchanged
        scale = torch.cat([norm_weights(wj, nj, hj.device) * nj, norm_weights(wr, nr, hr.device) * nr]).sqrt()
        x = x * scale[:, None]
    p = pca_basis(x, k)
    return p, mu_pool


def ot_targets(yj, yr, eps, target="ot", wj=None, wr=None):
    m_j, s_j = mean_cov(yj, eps, wj)
    m_r, s_r = mean_cov(yr, eps, wr)
    a_k, b_k = gaussian_ot_map(m_j, s_j, m_r, s_r)
    if target == "ot":
        delta = yj @ (a_k - torch.eye(a_k.shape[0], dtype=DTYPE, device=a_k.device)).T + b_k
    elif target == "constant":
        delta = (m_r - m_j).expand(yj.shape[0], -1).clone()
    else:
        raise ValueError(f"unknown target {target!r}")
    return delta, (a_k, b_k, m_j, s_j, m_r, s_r)


def selector_basis(hb, hr, hj, p, mu_pool, params: SelotParams):
    if params.selector == "y":
        return p, mu_pool
    h_all = torch.cat([hj, hr, hb])
    mu_all = h_all.mean(0)
    if params.selector == "full":
        return torch.eye(h_all.shape[1], dtype=DTYPE, device=h_all.device), mu_all
    if params.selector == "topm":
        return pca_basis(h_all - mu_all, params.m), mu_all
    raise ValueError(f"unknown selector {params.selector!r}")


def features(h, q, mu_all):
    return torch.cat([(h - mu_all) @ q, torch.ones(h.shape[0], 1, dtype=DTYPE, device=h.device)], dim=1)


def null_space_basis(psi_b, rel_tol, min_ratio):
    """Orthonormal basis (columns) of the small-eigenvalue eigenvectors of Psi_b^T Psi_b / n_b."""
    gram = sym(psi_b.T @ psi_b / psi_b.shape[0])
    evals, evecs = torch.linalg.eigh(gram)                         # ascending
    n_null = int((evals < rel_tol * evals.max()).sum())
    n_null = max(n_null, int(min_ratio * gram.shape[0]))
    return evecs[:, :n_null], evals


def fit_selot_layer(hb, hr, hj, params: SelotParams, wr=None, wj=None):
    """Returns (LayerMap with U, V, b0 and extras {P, mu_pool}, report dict). wr, wj: optional sample weights."""
    hb, hr, hj = as_f32(hb), as_f32(hr), as_f32(hj)
    nb, nr, nj = hb.shape[0], hr.shape[0], hj.shape[0]
    p, mu_pool = transport_subspace(hj, hr, params.k, wj, wr)
    yj, yr = (hj - mu_pool) @ p, (hr - mu_pool) @ p
    delta, (a_k, b_k, m_j, s_j, m_r, s_r) = ot_targets(yj, yr, params.eps, params.target, wj, wr)
    omega_j, omega_r = norm_weights(wj, nj, hj.device), norm_weights(wr, nr, hr.device)   # 1/n_j, 1/n_r if None
    report = {"k": p.shape[1], **check_ot_map(a_k, b_k, m_j, s_j, m_r, s_r)}

    q, mu_all = selector_basis(hb, hr, hj, p, mu_pool, params)
    psi_j, psi_r, psi_b = features(hj, q, mu_all), features(hr, q, mu_all), features(hb, q, mu_all)
    k = p.shape[1]
    if params.variant == "soft":
        feats = torch.cat([psi_j, psi_b, psi_r])
        targets = torch.cat([delta, torch.zeros(nb + nr, k, dtype=DTYPE, device=hj.device)])
        weights = torch.cat([omega_j, torch.full((nb,), params.beta / nb, device=hj.device),
                             params.beta_r * omega_r])
        m_star = weighted_ridge(feats, targets, weights, params.gamma)
    elif params.variant == "hard":
        e_null, evals = null_space_basis(psi_b, params.null_rel_tol, params.min_null_ratio)
        report["hard_null_dim"] = e_null.shape[1]
        report["hard_feature_dim"] = psi_b.shape[1]
        if e_null.shape[1] == 0:
            m_star = torch.zeros(k, psi_b.shape[1], dtype=DTYPE, device=hj.device)
        else:
            # M = Z E^T (so M N_b = M exactly); ||M||_F = ||Z||_F because E has orthonormal columns.
            feats = torch.cat([psi_j, psi_r]) @ e_null
            targets = torch.cat([delta, torch.zeros(nr, k, dtype=DTYPE, device=hj.device)])
            weights = torch.cat([omega_j, params.beta_r * omega_r])
            m_star = weighted_ridge(feats, targets, weights, params.gamma) @ e_null.T
    else:
        raise ValueError(f"unknown variant {params.variant!r}")

    w, c = m_star[:, :-1], m_star[:, -1]
    layer = LayerMap(U=p.contiguous(), V=(q @ w.T).contiguous(), b0=p @ (c - w @ (q.T @ mu_all)),
                     extras={"P": p, "mu_pool": mu_pool})
    report.update(fit_diagnostics(layer, hb, hr, hj, delta, m_star, (psi_b, psi_r, psi_j), params, omega_j, omega_r))
    return layer, report


def fit_diagnostics(layer, hb, hr, hj, delta, m_star, psis, params, omega_j, omega_r):
    """Training-set numbers the fit report shows (all at alpha = 1); jailbreak / refused averages use the sample
    weights omega (uniform without weights)."""
    psi_b, psi_r, psi_j = psis
    p, mu_pool = layer.extras["P"], layer.extras["mu_pool"]
    g_b, g_r, g_j = (layer.displacement(h) for h in (hb, hr, hj))
    # The affine (U, V, b0) form must reproduce P M psi(h) exactly.
    form_err = max(float((g - (m_star @ psi.T).T @ p.T).norm() / (g.norm() + 1e-12))
                   for g, psi in ((g_b, psi_b), (g_r, psi_r), (g_j, psi_j)))
    mean_sq = lambda g, w=None: float((g ** 2).sum(1).mean() if w is None else w @ (g ** 2).sum(1))
    delta_sq = mean_sq(delta, omega_j)
    mu_j, mu_r = omega_j @ hj, omega_r @ hr
    gap_before = p.T @ (mu_r - mu_j)
    gap_after = p.T @ (mu_r - omega_j @ (hj + g_j))
    rep = {
        "affine_form_rel_err": form_err,
        "g_sq_benign": mean_sq(g_b), "g_sq_refused": mean_sq(g_r, omega_r), "g_sq_jailbreak": mean_sq(g_j, omega_j),
        "delta_sq_jailbreak": delta_sq,
        "fit_rel_residual_jailbreak": mean_sq((m_star @ psi_j.T).T - delta, omega_j) / max(delta_sq, 1e-30),
        # Mean transport of the final field (approximate when beta, gamma > 0): fraction of the in-subspace
        # jailbreak->refusal mean gap left after steering, plus the gap outside the subspace (never moved).
        "field_mean_gap_left": float(gap_after.norm() / gap_before.norm().clamp_min(1e-12)),
        "mean_gap_outside_subspace": float(((mu_r - mu_j) - p @ gap_before).norm()
                                           / (mu_r - mu_j).norm().clamp_min(1e-12)),
    }
    if params.variant == "soft" and params.beta > 0:
        # Proposition 2: (1/n_b) sum ||g(h_b)||^2 <= (1/beta) sum_j omega_j ||delta*_j||^2  (omega_j = 1/n_j unweighted)
        rep["prop2_lhs"] = rep["g_sq_benign"]
        rep["prop2_rhs"] = delta_sq / params.beta
        rep["prop2_holds"] = rep["prop2_lhs"] <= rep["prop2_rhs"] * (1 + 1e-4) + 1e-8
    return rep
