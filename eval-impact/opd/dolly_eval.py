"""Instruction benchmark evaluation: batched generation, ROUGE-L and EM."""
import argparse
import importlib.metadata
import json
import random
import statistics
import string
import time
from pathlib import Path
import numpy as np
import torch
from .batching import rollout_inputs, trim_generated
from .data import digest, read_jsonl, save_json
from .recipes import adapter_signature


def normalize_answer(text):
    # Same normalization as the vendored DSKDv2 rouge_metric.py (keep articles).
    return " ".join("".join(c for c in text.lower() if c not in string.punctuation).split())


def score_response(prediction, references, scorer):
    references = references if isinstance(references, list) else [references]
    if not references or any(not isinstance(r, str) for r in references):
        raise ValueError("Evaluation references must be nonempty strings/list of strings")
    return {"rougeL": 100 * max(scorer.score(r, prediction)["rougeL"].fmeasure for r in references),
            "exact_match": 100 * float(any(normalize_answer(prediction) == normalize_answer(r) for r in references))}


def checkpoint_info(adapter, training_dir=None):
    info = {"checkpoint": str(Path(adapter).resolve()) if adapter else None,
            "adapter_sha256": adapter_signature(adapter) if adapter else None,
            "checkpoint_step": None, "training_complete": False, "training_state": "unknown"}
    if training_dir:
        root = Path(training_dir)
        latest = json.loads((root / "latest.json").read_text())
        if info["checkpoint"] != str(Path(latest["checkpoint"]).resolve()):
            raise ValueError("Selected evaluation adapter is not the latest checkpoint")
        status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        config = json.loads((root / "config.json").read_text())
        info.update(checkpoint_step=latest["step"], training_state=status.get("state", "unknown"),
                    training_steps_budget=config["steps"],
                    training_complete=status.get("state") == "completed" and
                        status.get("step") == latest["step"] and latest["step"] >= config["steps"])
        # Test labels must never have appeared among training examples.
        info["train_data"] = config["data"]
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--training-dir", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--benchmark", choices=["dolly", "self-inst", "vicuna", "sinst"], default="dolly")
    p.add_argument("--expected-samples", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Debug subset; never reported as the full test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    from .benchmarks import BENCHMARKS
    if args.expected_samples is None:
        args.expected_samples = 500 if args.benchmark == "dolly" else BENCHMARKS[args.benchmark]["n_samples"]
    if min(args.batch_size, args.max_prompt_length, args.max_new_tokens, args.expected_samples) < 1:
        raise ValueError("Batch and length budgets must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("Use distinct decoding seeds")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty eval output; the pipeline archives older evaluations automatically")
    from rouge_score.rouge_scorer import RougeScorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    rows = read_jsonl(args.data)
    if args.benchmark != "dolly":
        spec = BENCHMARKS[args.benchmark]
        data_manifest = json.loads(Path(args.data).with_name("manifest.json").read_text())
        if any(data_manifest.get(k) != spec[k] for k in ("repo", "revision", "file", "sha256")) or data_manifest.get("benchmark") != args.benchmark:
            raise ValueError("Benchmark source differs from the pinned reference")
        if data_manifest["prepared_sha256"] != digest(Path(args.data).read_text()):
            raise ValueError("Prepared benchmark checksum mismatch")
    if len(rows) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} test rows; got {len(rows)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test sample IDs")
    if any(not r.get("response") for r in rows):
        raise ValueError("Every evaluation sample needs a reference response")
    info = checkpoint_info(args.adapter, args.training_dir)
    if info.get("train_data"):
        train = read_jsonl(info["train_data"])
        if {r["prompt"] for r in train} & {r["prompt"] for r in rows}:
            raise ValueError("Train/test prompt overlap")
    rows = rows[:args.limit] if args.limit else rows
    expected = 500 if args.benchmark == "dolly" else BENCHMARKS[args.benchmark]["n_samples"]
    scope = "debug_subset" if args.limit else f"full_{args.benchmark}_test" if args.expected_samples == expected else "custom_test"
    protocol = {"data_sha256": digest(Path(args.data).read_text()), "n_samples": len(rows),
                "sample_ids_sha256": digest(json.dumps([r["id"] for r in rows])),
                "decoding_seeds": args.seeds, "batch_size": args.batch_size,
                "max_prompt_length": args.max_prompt_length, "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample, "temperature": 1., "top_p": 1., "top_k": 0,
                "prompt_format": "per-row raw/auto; prefix truncated to max_prompt_length",
                "dtype": args.dtype, "rouge_score_version": importlib.metadata.version("rouge-score"),
                "transformers_version": importlib.metadata.version("transformers"),
                "metric": "rougeL F1, use_stemmer=True; DSKDv2 normalized EM", "scope": scope}
    if args.benchmark != "dolly":
        protocol["benchmark"] = args.benchmark
    manifest = {"kind": args.benchmark, "state": "running", "protocol": protocol, **vars(args), **info,
                "scope": scope, "decoding_seed_note": "Decoding variability, not independent training seeds"}
    save_json(out / "evaluation_manifest.json", manifest)
    begin = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
              torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).merge_and_unload()
    model.to(args.device).eval().requires_grad_(False)
    vocab_ids = set(tok.get_vocab().values())
    suppressed = sorted(set(range(model.get_output_embeddings().weight.shape[0])) - vocab_ids)
    seed_scores = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        scores = []
        with (out / f"predictions-seed{seed}.jsonl").open("w") as stream:
            for start in range(0, len(rows), args.batch_size):
                selected = rows[start:start + args.batch_size]
                enc = rollout_inputs(tok, selected, args.device, args.max_prompt_length)
                sampling = {"temperature": 1., "top_p": 1., "top_k": 0} if args.do_sample else {}
                with torch.inference_mode():
                    output = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                        use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                        suppress_tokens=suppressed, **sampling)
                sampled = trim_generated(output[:, enc["input_ids"].shape[1]:], tok.eos_token_id)
                predictions = tok.batch_decode(sampled, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for row, prediction, ids in zip(selected, predictions, sampled):
                    score = score_response(prediction, row["response"], scorer)
                    scores.append(score)
                    stream.write(json.dumps({"id": row["id"], "seed": seed, "prompt": row["prompt"],
                        "reference": row["response"], "prediction": prediction, "token_ids": ids,
                        "n_tokens": len(ids), **score}, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({"seed": seed, "evaluated": len(scores), "total": len(rows)}), flush=True)
        seed_scores.append({"seed": seed, "n_samples": len(scores),
                            **{metric: statistics.mean(s[metric] for s in scores) for metric in ("rougeL", "exact_match")}})
    summary = {metric: {"mean": statistics.mean(s[metric] for s in seed_scores),
                       "std_decoding_seeds": statistics.stdev(s[metric] for s in seed_scores) if len(seed_scores) > 1 else None}
               for metric in ("rougeL", "exact_match")}
    result = {"kind": args.benchmark, "state": "completed", "protocol": protocol, **info,
              "per_seed": seed_scores, "summary": summary, "elapsed_seconds": time.monotonic() - begin}
    save_json(out / "results.json", result)
    save_json(out / "evaluation_manifest.json", {**manifest, "state": "completed"})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
