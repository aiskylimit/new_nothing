#!/usr/bin/env python3
"""One-time BI (Block Influence) calibration cost on a teacher model.

Reviewer request: report the one-time cost of computing BI scores on a
calibration set (forward + BI over hidden states).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_bi_scores(hidden_states):
    scores = []
    for l in range(1, len(hidden_states)):
        h_prev = hidden_states[l - 1].float()
        h_curr = hidden_states[l].float()
        cos = torch.nn.functional.cosine_similarity(h_prev, h_curr, dim=-1, eps=1e-6)
        scores.append(1.0 - cos.mean())
    return torch.stack(scores)


def load_prompts(data_dir: Path, max_samples: int):
    paths = [data_dir / "train.jsonl", data_dir / "train.txt", data_dir / "raw.txt"]
    prompts = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if p.suffix == ".jsonl":
                    obj = json.loads(line)
                    text = obj.get("prompt") or obj.get("instruction") or obj.get("input") or obj.get("text")
                    if text is None and "instruction" in obj:
                        text = obj["instruction"]
                        if obj.get("context"):
                            text = f"{text}\n{obj['context']}"
                    if text is None:
                        continue
                    prompts.append(str(text))
                else:
                    prompts.append(line)
                if len(prompts) >= max_samples:
                    return prompts
        if prompts:
            return prompts
    raise FileNotFoundError(f"No calibration prompts found under {data_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model-path", required=True)
    ap.add_argument("--teacher-peft-path", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_load0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.teacher_peft_path:
        model = PeftModel.from_pretrained(model, args.teacher_peft_path)
    model.to(device)
    model.eval()
    t_load1 = time.time()

    prompts = load_prompts(Path(args.data_dir), args.num_samples)
    n = len(prompts)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup 1 batch (excluded from timed BI cost)
    warm = tok(
        prompts[: min(args.batch_size, n)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(device)
    with torch.no_grad():
        _ = model(**warm, output_hidden_states=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    bi_acc = None
    n_batches = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            out = model(**enc, output_hidden_states=True)
            bi = compute_bi_scores(out.hidden_states)
            bi_acc = bi if bi_acc is None else bi_acc + bi
            n_batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    bi_mean = (bi_acc / max(n_batches, 1)).detach().float().cpu()
    topk = torch.topk(bi_mean, k=min(4, bi_mean.numel()))
    peak_mem_mib = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    )
    nvidia_used = None
    if device.type == "cuda":
        try:
            import os
            import subprocess

            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
            nvidia_used = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id",
                    gpu_id,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip().splitlines()[0]
        except Exception:
            nvidia_used = None

    result = {
        "num_samples": n,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_layers": int(bi_mean.numel()),
        "load_sec": round(t_load1 - t_load0, 3),
        "bi_calib_sec": round(t1 - t0, 3),
        "bi_calib_sec_per_sample": round((t1 - t0) / max(n, 1), 6),
        "peak_alloc_mib": round(peak_mem_mib, 1),
        "nvidia_smi_used_mib": int(nvidia_used) if nvidia_used is not None else None,
        "top4_layer_indices_0based": topk.indices.tolist(),
        "top4_bi_scores": [round(float(x), 6) for x in topk.values.tolist()],
        "note": (
            "One-time calibration: teacher forward(output_hidden_states)+BI over "
            f"{n} Dolly samples. In IMPACT training, BI is recomputed from already-"
            "available teacher hidden states each step (no extra teacher forward)."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
