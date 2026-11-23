"""Baselines expressed as the same per-layer affine map (selot.affine_map), plus the method registry.

    dim             difference in means  r = mean(Hr) - mean(H_source); A = I, b = alpha r        (Eq. 1-2)
    pca_ot          PCA-Gaussian OT, jailbreak -> refused, applied to every input (Prop. 1a)
    vector_file     any precomputed per-layer vector (e.g. Surgical, Jailbreak Antidote): A = I, b = alpha v
    alphasteer_form precomputed AlphaSteer matrix Delta (h <- h + lambda h Delta): A = I + lambda Delta^T, b = 0
    alphasteer      AlphaSteer fitted here on our fit sets, as its src/calc_steering_matrix.py + steering_utils.py:
                    P = null-space projection of H_b (the last int(d * ratio) right singular vectors of H_b^T H_b),
                    Delta~ = pinv(X^T X + lambda_reg P^T P) X^T (1 r^T) with X = H_h P, Delta = P Delta~;
                    H_b = D_b train, H_h = D_r train + D_j train (no weights), r = mean(D_b) - mean(D_r) (the
                    compliance direction: AlphaSteer steers with a negative strength); h <- h + strength h Delta,
                    i.e. A = I + strength Delta^T, b = 0, with steer.positions = alphasteer
    linear_act      per-neuron 1-D affine transport (Linear-AcT), jailbreak -> refused
    selot           the proposed method (selot.selot_fit)
    selot_gated     SELOT with a gate (Gaussian-mixture posterior or logistic probe) and per-mode Gaussian OT
                    instead of the ridge field (selot.gated_fit); not affine: the map lives in LayerMap.extras
"""

import torch

from selot.affine_map import AffineMap, LayerMap
from selot.gaussian_ot import check_ot_map
from selot.linalg import DTYPE, as_f32, wmean
from selot.gated_fit import GatedParams, fit_gated_layer
from selot.selot_fit import SelotParams, fit_selot_layer, ot_targets, transport_subspace


def fit_dim_layer(hb, hr, hj, source="benign", subspace_k=0, wr=None, wj=None):
    hr = as_f32(hr)
    if source == "jailbreak":
        mu_src = wmean(hj, wj)
    else:
        mu_src = as_f32({"benign": hb, "answered": torch.cat([hb, hj]) if hj is not None else None}[source]).mean(0)
    r = wmean(hr, wr) - mu_src
    report = {"dim_norm": float(r.norm())}
    extras = {}
    if subspace_k:                                   # Prop. 1b: P P^T (mu_r - mu_j)
        p, mu_pool = transport_subspace(as_f32(hj), hr, subspace_k, wj, wr)
        r = p @ (p.T @ r)
        extras = {"P": p, "mu_pool": mu_pool}
    return LayerMap(b0=r, extras=extras), report


def fit_pca_ot_layer(hb, hr, hj, k=5, eps=1e-4, wr=None, wj=None):
    hr, hj = as_f32(hr), as_f32(hj)
    p, mu_pool = transport_subspace(hj, hr, k, wj, wr)
    _, (a_k, b_k, m_j, s_j, m_r, s_r) = ot_targets((hj - mu_pool) @ p, (hr - mu_pool) @ p, eps, "ot", wj, wr)
    a_minus_i = a_k - torch.eye(a_k.shape[0], dtype=DTYPE, device=a_k.device)
    # g(h) = P[(A_k - I) P^T (h - mu_pool) + b_k]
    layer = LayerMap(U=p, V=p @ a_minus_i.T, b0=p @ (b_k - a_minus_i @ (p.T @ mu_pool)),
                     extras={"P": p, "mu_pool": mu_pool})
    report = {"k": p.shape[1], **check_ot_map(a_k, b_k, m_j, s_j, m_r, s_r)}
    # The lifted map must send the jailbreak mean onto the refusal mean inside the subspace.
    mu_j = wmean(hj, wj)
    y_after = p.T @ (layer.apply(mu_j[None], 1.0)[0] - mu_pool)
    report["lifted_mean_rel_err"] = float((y_after - m_r).norm() / m_r.norm().clamp_min(1e-12))
    return layer, report


def fit_linear_act_layer(hb, hr, hj, n_quantiles=0):
    """Per neuron i: least-squares line through the matched quantiles (sorted samples) of Hj[:, i] -> Hr[:, i]."""
    hr, hj = as_f32(hr), as_f32(hj)
    n_q = n_quantiles or min(hj.shape[0], hr.shape[0])
    q = torch.linspace(0, 1, n_q, dtype=DTYPE, device=hj.device)
    xs, ys = torch.quantile(hj, q, dim=0), torch.quantile(hr, q, dim=0)          # n_q x d
    xc, yc = xs - xs.mean(0), ys - ys.mean(0)
    slope = (xc * yc).sum(0) / (xc ** 2).sum(0).clamp_min(1e-12)
    intercept = ys.mean(0) - slope * xs.mean(0)
    return LayerMap(s=slope - 1.0, b0=intercept), {"linear_act_mean_slope": float(slope.mean())}


def alphasteer_null_projection(hb, ratio):
    """AlphaSteer's null_space_projection_l(H_b, abs_nullspace_ratio=ratio): P = Q Q^T, Q = the last int(d * ratio)
    right singular vectors of H_b^T H_b"""
    _, _, vh = torch.linalg.svd(hb.T @ hb)
    num = int(hb.shape[1] * ratio)
    q = vh[-num:, :].T
    return q @ q.T


def alphasteer_delta(hh, p, r, lambda_reg):
    """AlphaSteer's cal_tilde_delta_with_regularization_l + cal_steering_matrix_l: Delta = P pinv(A) X^T (1 r^T)"""
    x = hh @ p
    a = x.T @ x + lambda_reg * (p.T @ p)
    b = x.T @ r.repeat(x.shape[0], 1)
    return p @ (torch.linalg.pinv(a) @ b)


def fit_alphasteer_layer(hb, hr, hj, ratio, lambda_reg=10.0, strength=-0.5):
    hb, hr, hj = as_f32(hb), as_f32(hr), as_f32(hj)
    r = hb.mean(0) - hr.mean(0)
    p = alphasteer_null_projection(hb, ratio)
    hh = torch.cat([hr, hj])
    delta = alphasteer_delta(hh, p, r, lambda_reg)
    report = {"null_ratio": ratio, "null_dim": int(round(float(torch.trace(p)))), "r_norm": float(r.norm()),
              "delta_norm": float(delta.norm()),
              # how well the harmful prompts map onto r, and how little the benign ones move (AlphaSteer's goals)
              "harmful_fit_rel_err": float((hh @ delta - r).norm(dim=1).mean() / r.norm()),
              "benign_shift_rel": float((hb @ delta).norm(dim=1).mean() / r.norm())}
    return LayerMap(D=(strength * delta).T.contiguous()), report


def _per_layer_tensor(obj, layer):
    if isinstance(obj, dict):
        return obj[layer]
    return obj[layer]                                                             # [L, ...] tensor


def fit_map(method: dict, acts: dict, layers, device=None, weights=None) -> tuple:
    """method: the config's `method` block; acts: {"benign"|"refused"|"jailbreak": {layer: n x d}};
    weights: optional {"refused"|"jailbreak": n sample weights} (inverse topic frequency), used by selot, dim and
    pca_ot. linear_act matches unweighted quantiles.

    Returns (AffineMap, {layer: report}).
    """
    weights = weights or {}
    wr, wj = weights.get("refused"), weights.get("jailbreak")
    name = method["name"]
    layer_maps, reports = {}, {}
    hidden = None
    external = None
    if name in ("vector_file", "alphasteer_form"):
        external = torch.load(method["path"], map_location="cpu", weights_only=False)
    for layer in layers:
        hb, hr, hj = (acts[s][layer].to(device=device, dtype=DTYPE) if s in acts else None
                      for s in ("benign", "refused", "jailbreak"))
        if name == "selot":
            lm, rep = fit_selot_layer(hb, hr, hj, SelotParams.from_dict(method), wr, wj)
        elif name == "selot_gated":
            lm, rep = fit_gated_layer(hb, hr, hj, GatedParams.from_dict(method), wr, wj)
        elif name == "dim":
            lm, rep = fit_dim_layer(hb, hr, hj, method.get("source", "benign"), method.get("subspace_k", 0), wr, wj)
        elif name == "pca_ot":
            lm, rep = fit_pca_ot_layer(hb, hr, hj, method.get("k", 5), method.get("eps", 1e-4), wr, wj)
        elif name == "linear_act":
            lm, rep = fit_linear_act_layer(hb, hr, hj, method.get("n_quantiles", 0))
        elif name == "vector_file":
            lm, rep = LayerMap(b0=as_f32(torch.as_tensor(_per_layer_tensor(external, layer)))), {}
        elif name == "alphasteer":
            ratios = {int(k): float(v) for k, v in method["layers_ratio"].items()}
            if layer not in ratios:
                raise ValueError(f"alphasteer: no null-space ratio for layer {layer} (method.layers_ratio)")
            lm, rep = fit_alphasteer_layer(hb, hr, hj, ratios[layer], method.get("lambda_reg", 10.0),
                                           method.get("strength", -0.5))
        elif name == "alphasteer_form":
            delta = as_f32(torch.as_tensor(_per_layer_tensor(external, layer)))
            lm, rep = LayerMap(D=delta.T.contiguous()), {}                       # h Delta = Delta^T h
        else:
            raise ValueError(f"unknown method {name!r}")
        layer_maps[layer] = lm.to("cpu")
        reports[layer] = rep
        hidden = (next(iter(lm.terms().values())) if lm.terms() else lm.extras["P"]).shape[0]
    return AffineMap(name, hidden, layer_maps, meta={"method": method}), reports
