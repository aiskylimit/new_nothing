"""Save per-example BI, all layer ranks, uncertainty, and cross-corpus stability."""
import argparse
import csv
import itertools
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import digest, read_jsonl, save_json, write_jsonl


def masked_bi(hidden_states, attention_mask):
    """Return [batch, layers] means over valid tokens (equal-example aggregation)."""
    mask = attention_mask.bool()
    if (mask.sum(1) == 0).any():
        raise ValueError("BI requires at least one valid token per sample")
    scores = []
    for previous, current in zip(hidden_states[:-1], hidden_states[1:]):
        distance = 1 - F.cosine_similarity(previous.float(), current.float(), dim=-1, eps=1e-6)
        scores.append((distance * mask).sum(1) / mask.sum(1))
    return torch.stack(scores, dim=1)


def render_prompt(tokenizer, prompt):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt + "\n"


def summarize(samples, metadata, top_k=6, tau=2.0, bootstrap=1000, seed=42):
    if not samples or top_k < 1 or tau <= 0:
        raise ValueError("Nonempty samples, top_k > 0 and tau > 0 required")
    x = np.array([s["scores"] for s in samples], dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid BI observations")
    mean = x.mean(0)
    order = np.argsort(-mean, kind="stable")
    chosen = order[:min(top_k, len(mean))]
    w = np.exp((mean[chosen] - mean[chosen].max()) / tau)
    w /= w.sum()
    if len(x) > 1 and bootstrap > 0:
        rng = np.random.default_rng(seed)
        boot = np.stack([x[rng.integers(len(x), size=len(x))].mean(0) for _ in range(bootstrap)])
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        freq = np.zeros(len(mean))
        for b in boot:
            freq[np.argsort(-b, kind="stable")[:len(chosen)]] += 1 / bootstrap
    else:
        low, high, freq = mean, mean, np.full(len(mean), np.nan)
    ranks = np.argsort(order) + 1
    layers = []
    for i in range(len(mean)):
        layers.append({"layer": i + 1, "rank": int(ranks[i]), "bi_mean": float(mean[i]),
                       "bi_std": float(x[:, i].std(ddof=1)) if len(x) > 1 else None,
                       "ci95_low": float(low[i]), "ci95_high": float(high[i]),
                       "bootstrap_topk_frequency": float(freq[i]) if np.isfinite(freq[i]) else None,
                       "selected": bool(i in chosen),
                       "weight": float(w[list(chosen).index(i)]) if i in chosen else 0.0})
    return {"schema_version": 1, "metadata": metadata, "n_samples": len(x),
            "aggregation": "equal_example_mean_of_valid_token_BI", "layer_index_base": 1,
            "top_k": len(chosen), "tau": tau, "top_layers_ranked": (chosen + 1).tolist(),
            "layers": layers}


@torch.no_grad()
def collect(model, tokenizer, rows, out, model_id, max_length=1024, batch_size=1,
            top_k=6, tau=2.0, seed=42, text_mode="prompt", revision=None):
    out = Path(out)
    if not rows or batch_size < 1 or max_length < 1:
        raise ValueError("Nonempty calibration set and positive lengths/batch sizes required")
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    device = next(model.parameters()).device
    was_training = model.training
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            texts = [render_prompt(tokenizer, r["prompt"]) +
                     (r.get("response", "") if text_mode == "prompt_response" else "") for r in chunk]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                            add_special_tokens=False, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc, use_cache=False, output_hidden_states=True)
            values = masked_bi(outputs.hidden_states, enc["attention_mask"]).cpu().tolist()
            for j, (row, text, scores) in enumerate(zip(chunk, texts, values)):
                tokens = enc["input_ids"][j, enc["attention_mask"][j].bool()].tolist()
                samples.append({**row, "text_mode": text_mode, "rendered_text": text,
                                "input_ids": tokens, "n_tokens": len(tokens),
                                "input_ids_sha256": digest(str(tokens)), "scores": scores})
            del outputs
    finally:
        model.train(was_training)
        tokenizer.padding_side = old_padding
    config = getattr(model, "config", None)
    metadata = {"model": model_id, "revision": revision,
                "resolved_commit": getattr(config, "_commit_hash", None),
                "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                "max_length": max_length, "seed": seed, "text_mode": text_mode,
                "sample_ids": [r["id"] for r in rows],
                "inputs_sha256": digest("\n".join(s["input_ids_sha256"] for s in samples)),
                "hidden_state_convention": "HF output_hidden_states; terminal state may include final norm",
                "torch_version": torch.__version__,
                "layer_types": getattr(config, "layer_types", None)}
    result = summarize(samples, metadata, top_k=top_k, tau=tau, seed=seed)
    write_jsonl(out / "samples.jsonl", samples)
    save_json(out / "summary.json", result)
    with (out / "layers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["layers"][0]))
        writer.writeheader()
        writer.writerows(sorted(result["layers"], key=lambda x: x["rank"]))
    return result


def compare(a, b):
    for key in ("model", "revision", "resolved_commit", "tokenizer", "max_length", "text_mode", "hidden_state_convention"):
        if a["metadata"].get(key) != b["metadata"].get(key):
            raise ValueError(f"Cannot compare incompatible BI runs: {key}")
    la, lb = a["layers"], b["layers"]
    if len(la) != len(lb) or a["top_k"] != b["top_k"]:
        raise ValueError("Layer count/top_k must match")
    va = np.array([r["bi_mean"] for r in la])
    vb = np.array([r["bi_mean"] for r in lb])
    from scipy.stats import spearmanr, kendalltau
    rho, kendall = float(spearmanr(va, vb).statistic), float(kendalltau(va, vb).statistic)
    sa, sb = set(a["top_layers_ranked"]), set(b["top_layers_ranked"])
    return {"spearman": rho if math.isfinite(rho) else None,
            "kendall_tau_b": kendall if math.isfinite(kendall) else None,
            "topk_overlap_fraction": len(sa & sb) / a["top_k"],
            "topk_jaccard": len(sa & sb) / len(sa | sb),
            "mean_abs_score_change": float(np.abs(vb - va).mean()),
            "shared_sample_ids": len(set(a["metadata"]["sample_ids"]) & set(b["metadata"]["sample_ids"])),
            "top_layers_a": a["top_layers_ranked"], "top_layers_b": b["top_layers_ranked"],
            "per_layer": [{"layer": x["layer"], "bi_a": x["bi_mean"], "bi_b": y["bi_mean"],
                           "delta": y["bi_mean"] - x["bi_mean"],
                           "rank_a": x["rank"], "rank_b": y["rank"],
                           "rank_change": y["rank"] - x["rank"]} for x, y in zip(la, lb)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--revision", default=None)
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    c.add_argument("--max-length", type=int, default=1024)
    c.add_argument("--batch-size", type=int, default=1)
    c.add_argument("--top-k", type=int, default=6)
    c.add_argument("--tau", type=float, default=2)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--text-mode", choices=["prompt", "prompt_response"], default="prompt")
    d = sub.add_parser("compare")
    d.add_argument("--runs", nargs="+", required=True, help="LABEL=summary.json")
    d.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "collect":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                        torch_dtype=getattr(torch, args.dtype)).to(args.device)
        collect(model, tokenizer, read_jsonl(args.data), args.out, args.model,
                args.max_length, args.batch_size, args.top_k, args.tau, args.seed,
                args.text_mode, args.revision)
    else:
        import json
        runs = {name: json.loads(Path(path).read_text()) for name, path in
                (spec.split("=", 1) for spec in args.runs)}
        results = {f"{a} vs {b}": compare(runs[a], runs[b]) for a, b in itertools.combinations(runs, 2)}
        save_json(args.out, results)
        lines = ["# BI calibration stability", "", "Pair | Spearman | Kendall | Top-K overlap | Jaccard",
                 "--- | --- | --- | --- | ---"]
        for label, r in results.items():
            lines.append(f"{label} | {r['spearman']} | {r['kendall_tau_b']} | {r['topk_overlap_fraction']:.3f} | {r['topk_jaccard']:.3f}")
        Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
