#!/usr/bin/env python3
"""Stage `generate`: responses to every eval set at every strength alpha, with the map attached.

alpha = 0 is the unsteered model. Rows go to <work_dir>/generations/<run_id>/<set>.jsonl
({id, set, kind, metric, alpha, prompt, response | mmlu_pred, + the record's extra fields}); finished (alpha, id)
pairs are skipped, so the stage resumes after a timeout.

Per eval set (data.eval[*]): `metric` (selot.metrics), `template` (e.g. gsm8k), `max_new_tokens` and `max_samples`
override the `generate` block. Greedy decoding. MMLU is scored from the first answer token's logits, no generation.
Prompts are batched by length (less padding); `max_samples` takes a fixed random subset (seed 0).
Data parallel over GPUs: generate.shard = [i, n] (one process per GPU) answers every n-th record from i and writes
<set>.shard<i>of<n>.jsonl; merge_shards() folds those into <set>.jsonl afterwards (scripts/pipeline.py does). Rows
already in <set>.jsonl are skipped. generate.only_sets limits the stage to some eval sets.
"""

import json
import random

from common import read_jsonl, setup

import torch
from tqdm.auto import tqdm

from selot.activations import format_prompts, load_model, tokenize
from selot.affine_map import AffineMap
from selot.data import load_records
from selot.metrics import LETTERS, apply_template, letter_token_ids, mmlu_letter_scores, mmlu_prompt, set_kind, set_metric
from selot.steering import Steerer


def subset(records, n):
    if not n or n >= len(records):
        return records
    keep = set(random.Random(0).sample(range(len(records)), n))
    return [r for i, r in enumerate(records) if i in keep]


def shard_path(gen_dir, name, shard):
    return gen_dir / f"{name}.shard{shard[0]}of{shard[1]}.jsonl"


def merge_shards(gen_dir, name):
    """append the rows of <name>.shard*.jsonl that <name>.jsonl does not have yet, then delete the shard files"""
    out = gen_dir / f"{name}.jsonl"
    have = {(float(r["alpha"]), str(r["id"])) for r in read_jsonl(out)}
    added = 0
    for sp in sorted(gen_dir.glob(f"{name}.shard*of*.jsonl")):
        with open(out, "a") as f:
            for r in read_jsonl(sp):
                k = (float(r["alpha"]), str(r["id"]))
                if k not in have:
                    have.add(k)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    added += 1
        sp.unlink()
    return added


def still_valid(row, max_new, tok):
    """A stored response is reused when it was generated with this max_new_tokens, or with a smaller one that it did
    not reach (greedy decoding stopped at EOS, so a larger cap gives the same text). Rows without the field come from
    before it was recorded and are taken as generated with the current setting."""
    cap = row.get("max_new_tokens")
    if cap is None or cap == max_new:
        return True
    if cap < max_new:
        return len(tok(row.get("response", ""), add_special_tokens=False)["input_ids"]) < cap - 1
    return False


def main():
    cfg, paths, log = setup(__doc__)
    model, tok = load_model(cfg["model"]["path"], cfg["model"].get("dtype", "bfloat16"), cfg["device"])
    amap = AffineMap.load(paths["map"])
    steer_cfg, gen_cfg = cfg["steer"], cfg["generate"]
    steerer = Steerer(model, amap, steer_cfg.get("positions", "prompt"), steer_cfg.get("layers")).attach()
    log.info("steering %s at layers %s, positions=%s, alphas=%s", amap.method, steerer.layers,
             steerer.positions, steer_cfg["alphas"])
    paths["generations"].mkdir(parents=True, exist_ok=True)
    bs = gen_cfg["batch_size"]
    letters = None
    shard = gen_cfg.get("shard")
    only = set(gen_cfg.get("only_sets") or [])
    for spec in cfg["data"]["eval"]:
        if only and spec["name"] not in only:
            continue
        metric, kind = set_metric(spec), set_kind(spec)
        records = subset(load_records(spec["path"]), spec.get("max_samples", gen_cfg.get("max_samples")))
        max_new = spec.get("max_new_tokens", gen_cfg["max_new_tokens"])
        out = paths["generations"] / f"{spec['name']}.jsonl"
        merged = []
        if shard:
            i, n = int(shard[0]), int(shard[1])
            records = records[i::n]
            merged = [r for r in read_jsonl(out) if still_valid(r, max_new, tok) or metric == "mmlu"]
            out = shard_path(paths["generations"], spec["name"], (i, n))
        old = read_jsonl(out)
        if metric != "mmlu":
            keep = [r for r in old if still_valid(r, max_new, tok)]
            if len(keep) < len(old):
                log.info("%s: %d of %d responses were generated with another max_new_tokens and did not stop before "
                         "it: generated again with %d", spec["name"], len(old) - len(keep), len(old), max_new)
                out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
                old = keep
        done = {(float(r["alpha"]), str(r["id"])) for r in old + merged}
        for alpha in steer_cfg["alphas"]:
            todo = [r for r in records if (float(alpha), str(r["id"])) not in done]
            if not todo:
                log.info("skip %s alpha=%s: done", spec["name"], alpha)
                continue
            steerer.set_alpha(float(alpha))
            users = [mmlu_prompt(r) if metric == "mmlu" else apply_template(spec, r["prompt"]) for r in todo]
            texts = format_prompts(tok, users)
            order = sorted(range(len(todo)), key=lambda i: -len(texts[i]))
            with open(out, "a") as f:
                for i in tqdm(range(0, len(todo), bs), desc=f"{spec['name']} a={alpha}"):
                    idx = order[i:i + bs]
                    batch = tokenize(tok, [texts[j] for j in idx], model.device)
                    plen = batch["input_ids"].shape[1]
                    rows = []
                    if metric == "mmlu":
                        letters = letters or letter_token_ids(tok)
                        with steerer.prompt(plen), torch.no_grad():
                            logits = model(**batch, use_cache=False).logits[:, -1]
                        lp = mmlu_letter_scores(logits, letters).cpu()
                        for j, s in zip(idx, lp):
                            rows.append({"mmlu_pred": LETTERS[int(s.argmax())],
                                         "mmlu_logprobs": [round(float(x), 4) for x in s]})
                    else:
                        with steerer.prompt(plen), torch.no_grad():
                            gen = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                                                 temperature=None, top_p=None, top_k=None)
                        for resp in tok.batch_decode(gen[:, plen:], skip_special_tokens=True):
                            rows.append({"response": resp})
                    for j, extra in zip(idx, rows):
                        row = {**todo[j], "set": spec["name"], "kind": kind, "metric": metric,
                               "alpha": float(alpha), **extra}
                        if metric != "mmlu":
                            row["max_new_tokens"] = max_new
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
    steerer.detach()


if __name__ == "__main__":
    main()
