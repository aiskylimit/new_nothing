#!/usr/bin/env python3
"""Stage `label`: build this model's D_b, D_r, D_j from the candidate pools (selot/fit_data.py, steps iv-v).

1. The unsteered model answers every pooled prompt, greedy: fit_data.max_new_tokens.harmful for D_r / D_j pools,
   .benign for the benign pool (a refusal shows in the first tokens). `refused` = refusal prefix.
2. fit_data.behaviour = llamaguard: Llama Guard (fit_data.guard.path) labels every harmful-pool response unsafe / safe.
   keyword: unsafe = not a refusal. none: steps 1-2 are skipped and every pooled prompt is kept (ablation).
3. Keep, cap per topic, hold out 1/val_folds, weight by inverse topic frequency.
fit_data.engine = vllm runs step 1 with vLLM (scripts/vllm_worker.py, one subprocess, same chat template and token ids
as HF); hf uses transformers generate. Llama Guard (step 2) always runs on vLLM (selot.judge.LlamaGuardJudge).
Eval generation (steered and unsteered) always uses HF.
Labels: <work_dir>/behavior/<model.tag>/<set>.jsonl (shared by every run of the model; resumes; a row is redone when
its prompt changed). Sets: <work_dir>/fitsets/<model.tag>/<behaviour>/<set>.jsonl. Report: <results_dir>/fit_sets.json.
"""

import gc
import json

from common import setup, write_json

import torch
from tqdm.auto import tqdm

from selot.activations import format_prompts, load_model, tokenize
from selot.fit_data import (FIT_SETS, build_sets, data_cfg, pools, read_labels, unsafe_key, write_jsonl)
from selot.judge import LlamaGuardJudge, is_refusal
from selot.vllm_engine import vllm_generate


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


VLLM_CHUNK = 6000


def vllm_run(model_path, dtype, texts, ids, max_tokens, gpu_mem, log):
    """{id: output} from scripts/vllm_worker.py in a subprocess (frees the GPU when it exits)."""
    return vllm_generate(model_path, [str(i) for i in ids], max_tokens, dtype, gpu_mem, texts=texts, log=log)


def generate_vllm(cfg, dc, pool, label_dir, log):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"])
    for name in FIT_SETS:
        path = label_dir / f"{name}.jsonl"
        labels = read_labels(path)
        todo = [r for r in pool[name] if labels.get(r["id"], {}).get("prompt") != r["prompt"]]
        if not todo:
            log.info("responses %s: all %d done", name, len(pool[name]))
            continue
        max_new = dc["max_new_tokens"]["benign" if name == "benign" else "harmful"]
        for i in range(0, len(todo), VLLM_CHUNK):          # saved chunk by chunk: a stopped run keeps them
            part = todo[i:i + VLLM_CHUNK]
            resp = vllm_run(cfg["model"]["path"], cfg["model"].get("dtype", "bfloat16"),
                            format_prompts(tok, [r["prompt"] for r in part]), [r["id"] for r in part], max_new,
                            dc["vllm_gpu_mem"], log)
            with open(path, "a") as f:
                for r in part:
                    f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": resp[str(r["id"])],
                                        "refused": is_refusal(resp[str(r["id"])])}, ensure_ascii=False) + "\n")


def generate(cfg, dc, pool, label_dir, log):
    if dc["engine"] == "vllm":
        return generate_vllm(cfg, dc, pool, label_dir, log)
    model = tok = None
    for name in FIT_SETS:
        path = label_dir / f"{name}.jsonl"
        labels = read_labels(path)
        todo = [r for r in pool[name] if labels.get(r["id"], {}).get("prompt") != r["prompt"]]
        if not todo:
            log.info("responses %s: all %d done", name, len(pool[name]))
            continue
        if model is None:
            model, tok = load_model(cfg["model"]["path"], cfg["model"].get("dtype", "bfloat16"), cfg["device"])
        max_new = dc["max_new_tokens"]["benign" if name == "benign" else "harmful"]
        log.info("responses %s: %d prompts, %d new tokens", name, len(todo), max_new)
        texts = format_prompts(tok, [r["prompt"] for r in todo])
        order = sorted(range(len(todo)), key=lambda i: -len(texts[i]))
        bs = dc["batch_size"]
        with open(path, "a") as f:
            for i in tqdm(range(0, len(todo), bs), desc=f"label {name}"):
                idx = order[i:i + bs]
                batch = tokenize(tok, [texts[j] for j in idx], model.device)
                plen = batch["input_ids"].shape[1]
                with torch.no_grad():
                    gen = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                                         temperature=None, top_p=None, top_k=None)
                for j, resp in zip(idx, tok.batch_decode(gen[:, plen:], skip_special_tokens=True)):
                    f.write(json.dumps({"id": todo[j]["id"], "prompt": todo[j]["prompt"], "response": resp,
                                        "refused": is_refusal(resp)}, ensure_ascii=False) + "\n")
                f.flush()
    del model, tok
    free()


def judge(cfg, dc, label_dir, log):
    """unsafe labels of the harmful pools: one Llama Guard (vLLM) call over both sets"""
    behaviour, key = dc["behaviour"], unsafe_key(dc["behaviour"])
    labels = {name: read_labels(label_dir / f"{name}.jsonl") for name in ("refused", "jailbreak")}
    todo = [(name, l) for name, ls in labels.items() for l in ls.values() if key not in l]
    if not todo:
        return
    if behaviour == "keyword":
        verdicts = [not l["refused"] for _, l in todo]
    else:
        g = dc["guard"]
        log.info("Llama Guard (vLLM) on %d responses", len(todo))
        verdicts = LlamaGuardJudge(g["path"], g["dtype"], dc["vllm_gpu_mem"], log=log)([l for _, l in todo])
    for (_, l), u in zip(todo, verdicts):
        l[key] = bool(u)
    for name in {n for n, _ in todo}:
        write_jsonl(label_dir / f"{name}.jsonl", labels[name].values())


def main():
    cfg, paths, log = setup(__doc__)
    dc = data_cfg(cfg)
    pool = pools(cfg)
    log.info("behaviour %s; pools: %s", dc["behaviour"], {k: len(v) for k, v in pool.items()})
    labels = {name: {} for name in FIT_SETS}
    if dc["behaviour"] != "none":
        paths["behavior"].mkdir(parents=True, exist_ok=True)
        generate(cfg, dc, pool, paths["behavior"], log)
        judge(cfg, dc, paths["behavior"], log)
        labels = {name: read_labels(paths["behavior"] / f"{name}.jsonl") for name in FIT_SETS}
    sets, report = build_sets(cfg, pool, labels)
    for name, rows in sets.items():
        write_jsonl(paths["fitsets"] / f"{name}.jsonl", rows)
        s = report["sets"][name]
        log.info("%s: pool %d -> kept %d -> final %d (train %d, val %d); kept rate by source %s",
                 name, s["pool"], s["kept"], s["final"], s["train"], s["val"], s["kept_rate_by_source"])
        if s["train"] < 100:
            log.warning("%s has only %d train prompts for this model: the fitted moments will be noisy", name, s["train"])
    write_json(paths["results"] / "fit_sets.json", report)


if __name__ == "__main__":
    main()
