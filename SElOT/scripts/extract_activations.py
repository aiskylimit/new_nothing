#!/usr/bin/env python3
"""Stage `extract`: last-prompt-token activations of the three fitting sets at every fitted layer.

Writes <work_dir>/acts/<model.tag>/<set>.pt = {"layers", "acts": {layer: n x d float32}, "hash", "source"}.
A set is skipped when its file already holds the same prompts (hash) and all requested layers.
The prompts are the train rows of this model's fit sets (label stage, selot/fit_data.py); their sample weights
(inverse topic frequency) are stored next to the activations as "weights".
"""

from common import setup, write_json

import torch

from selot.activations import extract_last_token, load_model, num_layers, resolve_layers
from selot.data import prompts_hash
from selot.fit_data import FIT_SETS, fit_records


def main():
    cfg, paths, log = setup(__doc__)
    paths["acts"].mkdir(parents=True, exist_ok=True)
    model, tok = load_model(cfg["model"]["path"], cfg["model"].get("dtype", "bfloat16"), cfg["device"])
    layers = resolve_layers(cfg["model"], num_layers(model))
    log.info("layers: %s, fit sets: %s", layers, paths["fitsets"])
    only = set((cfg.get("extract") or {}).get("only_sets") or [])     # one process per GPU, e.g. [benign]
    for name in FIT_SETS:
        if only and name not in only:
            continue
        recs = fit_records(cfg, paths, name, "train")
        prompts = [r["prompt"] for r in recs]
        weights = torch.tensor([float(r.get("weight", 1.0)) for r in recs])
        out = paths["acts"] / f"{name}.pt"
        digest = prompts_hash(prompts + [repr(weights.tolist())])
        if out.exists():
            old = torch.load(out, map_location="cpu", weights_only=False)
            if old["hash"] == digest and set(layers) <= set(old["layers"]):
                log.info("skip %s: %s already has these %d prompts and layers", name, out, len(prompts))
                continue
        log.info("extract %s: %d prompts", name, len(prompts))
        acts = extract_last_token(model, tok, prompts, layers, cfg["extract"]["batch_size"])
        torch.save({"layers": layers, "acts": acts, "hash": digest, "source": str(paths["fitsets"] / f"{name}.jsonl"),
                    "n": len(prompts), "weights": weights}, out)
        log.info("saved %s (%d x %d per layer)", out, *acts[layers[0]].shape)
    write_json(paths["acts"] / "layers.json", {"layers": layers, "num_layers": num_layers(model)})


if __name__ == "__main__":
    main()
