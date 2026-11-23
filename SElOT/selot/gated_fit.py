"""SELOT-gated fit at one layer: a gate decides WHETHER to steer, Gaussian OT decides WHERE to.

SELOT's ridge field g = P M psi(h) has to be ~0 on benign activations (weight beta) and delta* on jailbreaks with one
linear map, so jailbreaks whose activations sit next to benign ones get almost no displacement, while far ones get
extrapolated past the refusal cluster. Here the two jobs are split:

    g(h) = P sum_c G_c(h_g) delta*_c(P^T (h - mu_pool)),   delta*_c(y) = (A_c - I) y + b_c

(a) transport subspace P, mu_pool: as SELOT (top-k PCA of D_j and D_r).
(b) gate features z = whitened top-gate_m PCs of the three sets, each set weighted 1/3 (benign does not dominate).
(c) gate (gate = gmm, default): a Gaussian mixture per set (D_j, D_b, D_r; full covariance, gate_reg added to the
    diagonal; components of each set by BIC up to max_components, or exactly max_components when gate_components =
    fixed. With BIC, K_j = 1 makes branch 3 the same map as branch 2 (one OT): the fit report's components /
    transport_components say which one was fitted);
    L(z) = log sum_{c in D_j} pi_c N_c(z) - log sum_{c in D_b, D_r} pi_c N_c(z) with set priors 1/2 (jailbreak) and
    1/2 split benign : refused = beta : beta_r. gate = logistic: L(z) = z w + b, a weighted logistic probe (same
    class weights). G = sigmoid((L - tau) / gate_temperature); tau is set so that a held-out part of D_b train
    (calib_frac, not used by the gate fit) gets G > 1/2 for a gate_fpr share only.
(d) transport: mixture = true (gmm gate): one Gaussian OT per jailbreak component, from the component (each jailbreak
    weighted by its responsibility) onto the refused set, and G_c = G * P(c | jailbreak, z): every mode of D_j gets
    the step it needs (the far template cluster is not pushed past the refusal cluster, the cloud next to benign gets
    a full step). mixture = false or logistic gate: one Gaussian OT for all of D_j (as SELOT's delta*), G_1 = G.
No ridge, so the field is not shrunk: a jailbreak the gate opens on gets the whole OT step (alpha = 1 lands on the
refused distribution inside the subspace).

At runtime the gate is computed once per prompt from the last prompt token (where it was fitted) and applied to every
prompt position (selot.steering, positions = prompt).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from selot.affine_map import LayerMap
from selot.gaussian_ot import check_ot_map, gaussian_ot_map
from selot.linalg import DTYPE, as_f32, mean_cov, norm_weights, pca_basis
from selot.selot_fit import transport_subspace


@dataclass
class GatedParams:
    k: int = 5                      # transport subspace size
    eps: float = 1e-4               # covariance shrinkage (transport)
    gate: str = "gmm"               # gmm | logistic
    mixture: bool = True            # gmm gate: one OT per jailbreak component
    gate_m: int = 64                # gate feature size (whitened PCs)
    max_components: int = 4         # gmm: per set (D_j, D_b, D_r)
    gate_components: str = "bic"    # bic | fixed (always max_components)
    gate_reg: float = 1e-2          # gmm: added to every covariance diagonal (whitened units)
    gate_C: float = 1.0             # logistic: inverse L2 strength
    gate_fpr: float = 0.01          # held-out benign share with G > 1/2; <= 0: tau = 0
    gate_temperature: float = 1.0
    calib_frac: float = 0.2         # share of D_b train held out for tau
    beta: float = 100.0             # negative prior split benign : refused = beta : beta_r
    beta_r: float = 1.0
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "GatedParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def gate_basis(hb, hr, hj, m):
    """center, d x m projection to whitened top-m PCs; each set weighs 1/3"""
    parts = [hj, hr, hb]
    mu = sum(p.mean(0) for p in parts) / 3
    x = torch.cat([(p - mu) / math.sqrt(3 * p.shape[0]) for p in parts])
    q = pca_basis(x, m)
    z = x @ q                                       # weighted rows: column variance = sum of squares
    scale = z.pow(2).sum(0).sqrt().clamp_min(1e-8)
    return mu, q / scale


def priors(params):
    wb = params.beta / (params.beta + params.beta_r)
    return {0: 0.5, 1: 0.5 * wb, 2: 0.5 * (1 - wb)}


def fit_gmm(z: np.ndarray, params: GatedParams):
    from sklearn.mixture import GaussianMixture
    ks = [params.max_components] if params.gate_components == "fixed" else range(1, params.max_components + 1)
    best = None
    for k in ks:
        if k > 1 and len(z) < 30 * k:
            break
        g = GaussianMixture(k, covariance_type="full", reg_covar=params.gate_reg, n_init=2, max_iter=300,
                            random_state=params.seed).fit(z)
        bic = g.bic(z)
        if best is None or bic < best[0]:
            best = (bic, g)
    return best[1]


def fit_gated_layer(hb, hr, hj, params: GatedParams, wr=None, wj=None):
    """Returns (gated LayerMap, report). wr, wj: optional sample weights of D_r / D_j (transport only)."""
    if params.gate not in ("gmm", "logistic"):
        raise ValueError(f"unknown gate {params.gate!r}")
    hb, hr, hj = as_f32(hb), as_f32(hr), as_f32(hj)
    dev = hj.device
    p, mu_pool = transport_subspace(hj, hr, params.k, wj, wr)
    k = p.shape[1]
    yj, yr = (hj - mu_pool) @ p, (hr - mu_pool) @ p

    # gate features; part of D_b train held out for the threshold
    g_mu, g_q = gate_basis(hb, hr, hj, params.gate_m)
    m = g_q.shape[1]
    perm = torch.randperm(hb.shape[0], generator=torch.Generator().manual_seed(params.seed)).to(dev)
    n_cal = max(1, round(params.calib_frac * hb.shape[0])) if params.gate_fpr > 0 else 0
    hb_cal, hb_fit = hb[perm[:n_cal]], hb[perm[n_cal:]]
    feat = lambda h: ((h - g_mu) @ g_q).double().cpu().numpy()
    zj, zb, zr = feat(hj), feat(hb_fit), feat(hr)
    pri = priors(params)
    extras = {"P": p, "mu_pool": mu_pool, "gate_mu": g_mu, "gate_Q": g_q}
    report = {"k": k, "gate": params.gate, "gate_m": m, "calib_benign": int(n_cal)}

    resp = None
    if params.gate == "gmm":
        mix = {0: fit_gmm(zj, params), 1: fit_gmm(zb, params), 2: fit_gmm(zr, params)}
        means, chols, logc, cls = [], [], [], []
        for c, g in mix.items():
            for i in range(g.n_components):
                ch = g.precisions_cholesky_[i]
                means.append(g.means_[i])
                chols.append(ch)
                logc.append(math.log(pri[c]) + math.log(g.weights_[i]) - 0.5 * m * math.log(2 * math.pi)
                            + float(np.log(np.diag(ch)).sum()))
                cls.append(c)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=DTYPE, device=dev)
        extras.update(gate_means=t(means), gate_chol=t(chols), gate_logc=t(logc), gate_class=t(cls))
        report["components"] = {"jailbreak": mix[0].n_components, "benign": mix[1].n_components,
                                "refused": mix[2].n_components}
        if params.mixture and mix[0].n_components > 1:
            resp = torch.as_tensor(mix[0].predict_proba(zj), dtype=DTYPE, device=dev)       # n_j x C
    else:
        from sklearn.linear_model import LogisticRegression
        X = np.concatenate([zj, zb, zr])
        y = np.r_[np.ones(len(zj)), np.zeros(len(zb) + len(zr))]
        w = np.r_[np.full(len(zj), pri[0] / len(zj)), np.full(len(zb), pri[1] / len(zb)),
                  np.full(len(zr), pri[2] / len(zr))]
        lr = LogisticRegression(C=params.gate_C, max_iter=5000).fit(X, y, sample_weight=w * len(w))
        extras.update(gate_w=torch.as_tensor(lr.coef_[0], dtype=DTYPE, device=dev),
                      gate_b=torch.as_tensor(lr.intercept_, dtype=DTYPE, device=dev))

    # transport: one Gaussian OT per jailbreak component (or one for all of D_j)
    m_r, s_r = mean_cov(yr, params.eps, wr)
    base = norm_weights(wj, yj.shape[0], dev)
    comps = [base] if resp is None else [base * resp[:, c] for c in range(resp.shape[1])]
    eye = torch.eye(k, dtype=DTYPE, device=dev)
    ot_a, ot_b, errs, n_eff = [], [], [], []
    for wc in comps:
        m_c, s_c = mean_cov(yj, params.eps, wc)
        a_c, b_c = gaussian_ot_map(m_c, s_c, m_r, s_r)
        errs.append(check_ot_map(a_c, b_c, m_c, s_c, m_r, s_r))
        ot_a.append(a_c - eye)
        ot_b.append(b_c)
        n_eff.append(float(wc.sum() ** 2 / (wc ** 2).sum()))
    extras.update(ot_A=torch.stack(ot_a), ot_b=torch.stack(ot_b),
                  gate_tau=torch.zeros((), dtype=DTYPE, device=dev),
                  gate_temp=torch.tensor(float(params.gate_temperature), dtype=DTYPE, device=dev))
    layer = LayerMap(extras=extras)
    if n_cal:
        logit_cal = layer.gate(hb_cal)[0]
        extras["gate_tau"] = torch.quantile(logit_cal, 1 - params.gate_fpr).to(DTYPE)
    for key in ("ot_mean_rel_err", "ot_cov_rel_err", "ot_A_asym"):
        report[key] = max(e[key] for e in errs)
    report["ot_A_min_eig"] = min(e["ot_A_min_eig"] for e in errs)
    report.update(transport_components=len(comps), transport_n_eff=[round(x, 1) for x in n_eff],
                  gate_tau=float(extras["gate_tau"]))
    report.update(gate_diagnostics(layer, {"jailbreak": hj, "benign_fit": hb_fit, "benign_calib": hb_cal,
                                           "refused": hr}))
    g_j = layer.displacement(hj)
    mu_j, mu_r = base @ hj, norm_weights(wr, hr.shape[0], dev) @ hr
    gap_before = p.T @ (mu_r - mu_j)
    report["field_mean_gap_left"] = float((p.T @ (mu_r - base @ (hj + g_j))).norm() / gap_before.norm().clamp_min(1e-12))
    report["mean_gap_outside_subspace"] = float(((mu_r - mu_j) - p @ gap_before).norm()
                                                / (mu_r - mu_j).norm().clamp_min(1e-12))
    return layer, report


def gate_diagnostics(layer, sets):
    """per set: mean gate, share with gate > 1/2, mean ||g||^2 at alpha = 1"""
    out = {}
    for name, h in sets.items():
        if h.shape[0] == 0:
            continue
        _, g, _ = layer.gate(h)
        out[f"gate_mean_{name}"] = float(g.mean())
        out[f"gate_open_{name}"] = float((g > 0.5).float().mean())
        out[f"g_sq_{name}"] = float(layer.displacement(h).pow(2).sum(1).mean())
    return out
