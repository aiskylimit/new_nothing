#!/usr/bin/env python3
"""Stage `fit`: fit the configured method's affine map at every layer, check it numerically, save it.

Writes <work_dir>/maps/<run_id>/map.pt and <results_dir>/fit_report.json. Exits non-zero when an exact
identity fails (OT mean/covariance transport, affine form == P M psi(h), lifted mean, Proposition 2).
"""

import json

from common import setup, write_json

import torch

from selot.baselines import fit_map

# Identities that hold exactly up to float32 round-off; anything above these tolerances is a bug.
CHECKS = {
    "ot_mean_rel_err": 1e-3,        # A_k m_j + b_k = m_r
    "ot_cov_rel_err": 1e-2,         # A_k S_j A_k = S_r
    "affine_form_rel_err": 1e-3,    # U V^T h + b0 = P M psi(h)
    "lifted_mean_rel_err": 1e-3,    # PCA-OT: P^T(T(mu_j) - mu_pool) = m_r
}


def check(reports):
    failures = []
    for layer, rep in reports.items():
        for key, tol in CHECKS.items():
            if key in rep and not rep[key] <= tol:
                failures.append(f"layer {layer}: {key}={rep[key]:.3g} > {tol}")
        if rep.get("prop2_holds") is False:
            failures.append(f"layer {layer}: Proposition 2 violated ({rep['prop2_lhs']:.4g} > {rep['prop2_rhs']:.4g})")
    return failures


def main():
    cfg, paths, log = setup(__doc__)
    method = cfg["method"]
    acts, weights = {}, {}
    for name in ("benign", "refused", "jailbreak"):
        f = paths["acts"] / f"{name}.pt"
        if f.exists():
            saved = torch.load(f, map_location="cpu", weights_only=False)
            acts[name] = saved["acts"]
            if name != "benign" and saved.get("weights") is not None and not bool((saved["weights"] == 1).all()):
                weights[name] = saved["weights"]
    if not acts:
        raise SystemExit(f"no activations under {paths['acts']}: run the extract stage first")
    layers = json.loads((paths["acts"] / "layers.json").read_text())["layers"]
    if cfg["model"].get("layers"):
        absent = sorted(set(cfg["model"]["layers"]) - set(layers))
        if absent:
            raise SystemExit(f"model.layers {absent} have no activations under {paths['acts']} (extracted: {layers}): "
                             "run the extract stage at these layers")
        layers = [l for l in layers if l in set(cfg["model"]["layers"])]
    log.info("fit %s at layers %s (sizes: %s; topic-weighted: %s)", method["name"], layers,
             {k: tuple(v[layers[0]].shape) for k, v in acts.items()}, sorted(weights))
    amap, reports = fit_map(method, acts, layers, device=cfg.get("fit_device", cfg["device"]), weights=weights)
    paths["map"].parent.mkdir(parents=True, exist_ok=True)
    amap.save(paths["map"])
    failures = check(reports)
    write_json(paths["results"] / "fit_report.json",
               {"method": method, "layers": layers, "map": str(paths["map"]), "checks": CHECKS,
                "failures": failures, "per_layer": reports})
    for layer, rep in reports.items():
        log.info("layer %d: %s", layer, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rep.items()})
    if failures:
        raise SystemExit("numerical checks FAILED:\n  " + "\n  ".join(failures))
    log.info("saved %s; all numerical checks passed", paths["map"])


if __name__ == "__main__":
    main()
