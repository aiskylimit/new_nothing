#!/usr/bin/env python3
"""Stage `analyze`: activation-level checks of requirements (i) and (ii) (proposal Sec. 4.4).

RQ3: displacement norms alpha * ||g(h)|| for each set in `analysis.sets` (benign, borderline, refused,
     jailbreak prompts not used for fitting), at the last prompt token and over every prompt position.
(i): Gaussian W2 / covariance alignment of the fitting jailbreaks vs refused, before and after steering.
Writes <results_dir>/analysis.json.
"""

from common import setup, write_json

import torch

from selot.activations import extract_all_tokens, load_model
from selot.affine_map import AffineMap
from selot.analysis import displacement_stats, transport_match
from selot.data import load_prompts


def main():
    cfg, paths, log = setup(__doc__)
    amap = AffineMap.load(paths["map"])
    layers = cfg["steer"].get("layers") or sorted(amap.layers)
    alphas = [a for a in cfg["steer"]["alphas"] if float(a) != 0.0] or [1.0]
    result = {"method": amap.method, "layers": layers, "alphas": alphas, "transport": {}, "displacement": {}}

    fit = {}
    for name in ("refused", "jailbreak"):
        f = paths["acts"] / f"{name}.pt"
        if f.exists():
            fit[name] = torch.load(f, map_location="cpu", weights_only=False)["acts"]
    if len(fit) == 2:
        for l in layers:
            result["transport"][l] = {str(a): transport_match(amap.layers[l], fit["jailbreak"][l], fit["refused"][l], float(a))
                                      for a in alphas}

    sets = cfg.get("analysis", {}).get("sets", [])
    if sets:
        model, tok = load_model(cfg["model"]["path"], cfg["model"].get("dtype", "bfloat16"), cfg["device"])
        for spec in sets:
            prompts = load_prompts(spec["path"])[: cfg.get("analysis", {}).get("max_samples") or None]
            acts, is_last = extract_all_tokens(model, tok, prompts, layers, cfg["extract"]["batch_size"])
            entry = {}
            seq = torch.cumsum(is_last.long(), 0) - is_last.long()      # prompt index of every token row
            for l in layers:
                lm = amap.layers[l]
                gate_h = acts[l][is_last][seq] if lm.gated else None      # runtime: last-token gate at all positions
                entry[l] = {str(a): {"last_token": displacement_stats(lm, acts[l][is_last], float(a)),
                                     "all_prompt_tokens": displacement_stats(lm, acts[l], float(a), gate_h)}
                            for a in alphas}
            result["displacement"][spec["name"]] = {"group": spec.get("group", spec["name"]), "per_layer": entry}
            log.info("%s: layer %d alpha %s last-token mean ||disp|| = %.4g", spec["name"], layers[0], alphas[-1],
                     entry[layers[0]][str(alphas[-1])]["last_token"]["mean"])
    write_json(paths["results"] / "analysis.json", result)


if __name__ == "__main__":
    main()
